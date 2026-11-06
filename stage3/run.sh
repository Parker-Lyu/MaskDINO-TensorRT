#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: bash stage3/run.sh DATASET_ROOT RUN_ID COMMAND [ARGS...]" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dataset=$(realpath -- "$1")
run_id=$2
shift 2
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
cd "$root"
[[ -z "$(git status --porcelain)" ]] || { echo "Run from a clean checkout" >&2; exit 1; }
revision=$(git rev-parse HEAD)
image=$(docker image inspect --format '{{.Id}}' "${MASKDINO_IMAGE:-maskdino:stage1}")
checkpoint=$(realpath -- "${MASKDINO_CHECKPOINT:-$root/weights/maskdino_r50_panoptic.pth}")
output="$root/stage3/runs/$run_id"
mkdir -p "$(dirname "$output")"
mkdir "$output"
{
    echo "Project commit: $revision"
    echo "Container image: $image"
    printf 'Command:'; printf ' %q' "$@"; echo
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    printf '%s  %s\n' 3e3d04a930ad42e14e97bc68d0e5844eefde702aa3d0a2324e603e5e9b5d0d37 "$checkpoint" | sha256sum -c
    printf '%s  %s\n' 454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a "$dataset/annotations/panoptic_val2017.json" | sha256sum -c
    sha256sum "$checkpoint" "$dataset/annotations/panoptic_val2017.json" stage2/sample.json
    docker run --rm --gpus all --ipc=host --network=none \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=1 \
        -e DETECTRON2_DATASETS=/coco -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        --mount "type=bind,source=$dataset,target=/coco,readonly" \
        --mount "type=bind,source=$checkpoint,target=/checkpoint.pth,readonly" \
        --mount "type=bind,source=$root,target=/project,readonly" \
        --mount "type=bind,source=$output,target=/run-output" \
        --entrypoint bash "$image" -ec '
            revision=$(git -c safe.directory="$PWD" rev-parse HEAD)
            echo "MaskDINO commit: $revision"
            [[ "$revision" == 3831d8514a3728535ace8d4ecc7d28044c42dd14 ]]
            export PYTHONPATH="/project:$PWD${PYTHONPATH:+:$PYTHONPATH}"
            if [[ "$1" == trace ]]; then
                shift
                exec nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
                    --capture-range=cudaProfilerApi --capture-range-end=stop --output=/run-output/trace \
                    python -u -m stage3.run trace "$@"
            fi
            if [[ "$1" == repro-constants ]]; then
                shift
                exec python -u -m stage3.repro_constants "$@"
            fi
            exec python -u -m stage3.run "$@"
        ' bash "$@"
} 2>&1 | tee "$output/console.log"
