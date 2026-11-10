#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: bash stage3/profile_layers.sh DATASET_ROOT B1_ENGINE RUN_ID" >&2
    exit 2
fi
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=$PWD
dataset=$(realpath -- "$1")
engine=$(realpath -- "$2")
run_id=$3
[[ "$run_id" =~ ^[[:alnum:]][[:alnum:]_-]*$ ]]
[[ -z "$(git status --porcelain)" ]] || { echo "Run from a clean checkout" >&2; exit 1; }
image=$(docker image inspect --format '{{.Id}}' "${MASKDINO_IMAGE:-maskdino:stage1}")
output="$root/stage3/runs/$run_id"
mkdir -p "$(dirname "$output")"
mkdir "$output"

{
    echo "Project commit: $(git rev-parse HEAD)"
    echo "Container image: $image"
    sha256sum "$engine" "$dataset/annotations/panoptic_val2017.json" stage2/sample.json
    nvidia-smi
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
        echo "GPU already has a compute process" >&2
        exit 1
    fi

    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem \
        --format=csv -l 1 > "$output/gpu-samples.csv" &
    gpu_monitor=$!
    nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_gpu_memory --format=csv -l 1 \
        > "$output/compute-processes.csv" &
    process_monitor=$!
    trap 'kill "$gpu_monitor" "$process_monitor" 2>/dev/null || true' EXIT

    docker run --rm --gpus all --network=none --user "$(id -u):$(id -g)" -e HOME=/tmp \
        --mount "type=bind,source=$root,target=/project,readonly" \
        --mount "type=bind,source=$dataset,target=/coco,readonly" \
        --mount "type=bind,source=$engine,target=/network.engine,readonly" \
        --mount "type=bind,source=$output,target=/run-output" \
        --entrypoint bash "$image" -euo pipefail -c '
            nvcc --version
            dpkg-query -W libnvinfer10 libjpeg-turbo8
            cd /project/stage4
            nvcc -std=c++17 -O3 -arch=sm_89 --fmad=false -Xcompiler=-fPIC,-ffp-contract=off \
                --shared preprocess.cu preprocess_check.cu -ljpeg -o /run-output/preprocess-check.so
            cd /run-output
            python /project/stage3/profile_inputs.py /coco 149770 456394 107339

            common=(--loadEngine=/network.engine --infStreams=1 --noDataTransfers
                    --warmUp=2000 --duration=5 --iterations=300 --percentile=50,95)
            for sample in 149770-r1 149770-r2 149770-r3 456394-r1 107339-r1; do
                id=${sample%-r*}
                echo "Profiling $sample"
                trtexec "${common[@]}" --loadInputs="image:$id.f32" --separateProfileRun \
                    --dumpProfile --exportProfile="$sample.layers.json" \
                    --exportTimes="$sample.times.json" > "$sample.log" 2>&1
            done
        '
    nvidia-smi
} 2>&1 | tee "$output/console.log"
