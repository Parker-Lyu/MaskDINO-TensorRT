#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
    echo "Usage: bash stage4/run.sh DATASET ENGINE RUN_ID native|reference|pytorch check|align|evaluate|benchmark|trace|trace-stage-waits" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dataset=$(realpath -- "$1")
engine=$(realpath -- "$2")
id=$3
backend=$4
mode=$5
[[ "$id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
case "$backend:$mode" in
    native:check|native:align|native:evaluate|native:benchmark|native:trace|native:trace-stage-waits|reference:benchmark|reference:trace|pytorch:benchmark) ;;
    *) echo "Unsupported backend and mode" >&2; exit 2 ;;
esac
cd "$root"
[[ -z "$(git status --porcelain)" ]] || { echo "Run from a clean checkout" >&2; exit 1; }
image=$(docker image inspect --format '{{.Id}}' "${MASKDINO_IMAGE:-maskdino:stage1}")
checkpoint=$(realpath -- "${MASKDINO_CHECKPOINT:-$root/weights/maskdino_r50_panoptic.pth}")
build="$(dirname -- "$engine")/build.json"
engine_sha=96facc4e2ee6e2bb61b2a55ae634447a6a019b3c90ef5bb88049143c523cb37a
build_mount=()
precision=fp32
if [[ -f "$build" ]]; then
    engine_sha=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["engine_sha256"])' "$build")
    precision=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["precision"])' "$build")
    build_mount=(--mount "type=bind,source=$build,target=/build.json,readonly")
fi
output="$root/stage4/runs/$id"
mkdir -p "$(dirname "$output")"
mkdir "$output"
{
    echo "Project commit: $(git rev-parse HEAD)"
    echo "Container image: $image"
    printf 'Command:'; printf ' %q' "$@"; echo
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    printf '%s  %s\n' "$engine_sha" "$engine" | sha256sum -c
    printf '%s  %s\n' 3e3d04a930ad42e14e97bc68d0e5844eefde702aa3d0a2324e603e5e9b5d0d37 "$checkpoint" | sha256sum -c
    printf '%s  %s\n' 454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a "$dataset/annotations/panoptic_val2017.json" | sha256sum -c
    sha256sum "$engine" "$checkpoint" "$dataset/annotations/panoptic_val2017.json" stage2/sample.json
    docker run --rm --gpus all --pid=host --ipc=host --network=none \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=1 \
        -e DETECTRON2_DATASETS=/coco -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        --mount "type=bind,source=$dataset,target=/coco,readonly" \
        --mount "type=bind,source=$checkpoint,target=/checkpoint.pth,readonly" \
        --mount "type=bind,source=$engine,target=/network.engine,readonly" \
        "${build_mount[@]}" \
        --mount "type=bind,source=$root,target=/project,readonly" \
        --mount "type=bind,source=$output,target=/run-output" \
        --entrypoint bash "$image" -ec '
            backend=$1
            mode=$2
            native_mode=$mode
            if [[ "$mode" == trace-stage-waits ]]; then mode=trace; fi
            echo "MaskDINO commit: $(git -c safe.directory="$PWD" rev-parse HEAD)"
            [[ "$(git -c safe.directory="$PWD" rev-parse HEAD)" == 3831d8514a3728535ace8d4ecc7d28044c42dd14 ]]
            export PYTHONPATH="/project:$PWD${PYTHONPATH:+:$PYTHONPATH}"
            output=/run-output
            profile=(nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
                     --capture-range=cudaProfilerApi --capture-range-end=stop --output="$output/trace")
            if [[ "$backend" != native ]]; then
                args=(python -u -m stage3.run "$mode" --backend "$3" --engine /network.engine)
                if [[ "$backend" == pytorch ]]; then args=(python -u -m stage3.run "$mode" --backend pytorch); fi
                if [[ "$mode" == trace ]]; then "${profile[@]}" "${args[@]}"; else "${args[@]}" --process-memory; fi
            else
                nvcc --version
                dpkg-query -W libnvinfer10 libjpeg-turbo8
                bash /project/stage4/build.sh "$output"
                ldd "$output/maskdino-native" | tee "$output/runtime-libraries.txt"
                if [[ "$mode" == check ]]; then
                    python -m stage4.check_preprocess "$output/preprocess-check.so" /coco | tee "$output/preprocess.jsonl"
                    python -m stage4.check_restore "$output/restore-check.so" | tee "$output/restore.jsonl"
                    exit
                fi
                python -m stage4.protocol manifest "$mode" "$output" /network.engine
                if [[ "$mode" == align || "$mode" == evaluate ]]; then native_mode=predict; fi
                args=("$output/maskdino-native" /network.engine "$output/things.txt" "$output/manifest.tsv" "$native_mode" "$output")
                if [[ "$mode" == trace ]]; then "${profile[@]}" "${args[@]}"; else "${args[@]}"; fi
                python -u -m stage4.protocol report "$mode" "$output" /network.engine
            fi
            if [[ "$mode" == trace ]]; then
                nsys export --type=sqlite --output="$output/trace.sqlite" "$output/trace.nsys-rep"
                nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
                    "$output/trace.sqlite" > "$output/nsys-stats.txt"
                nsys stats --report nvtx_gpu_proj_sum,nvtx_gpu_proj_trace \
                    "$output/trace.sqlite" > "$output/nsys-gpu-projection.txt"
                python -m stage3.trace_summary "$output/trace.sqlite" > "$output/trace-summary.json"
                (cd "$output"; sha256sum trace.nsys-rep > trace.sha256)
            fi
        ' bash "$backend" "$mode" "$precision"
} 2>&1 | tee "$output/console.log"
