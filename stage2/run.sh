#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
    echo "Usage: bash stage2/run.sh DATASET_ROOT {probe|evaluate|benchmark} SIZE BATCH" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dataset=$(realpath -- "$1")
mode=$2
size=$3
batch=$4
[[ "$mode" =~ ^(probe|evaluate|benchmark)$ && "$size" =~ ^(640|800|1024)$ && "$batch" =~ ^(1|2|4|8)$ ]]
image=${MASKDINO_IMAGE:-maskdino:stage1}
checkpoint=$(realpath -- "${MASKDINO_CHECKPOINT:-$root/weights/maskdino_r50_panoptic.pth}")
revision=$(git -C "$root" rev-parse HEAD)
[[ -z "$(git -C "$root" status --porcelain)" ]] || { echo "Run from a clean checkout" >&2; exit 1; }
run_id=${MASKDINO_RUN_ID:-$revision}
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
output="$root/stage2/runs/$run_id/$mode/$size-$batch"
mkdir -p "$(dirname "$output")"
mkdir "$output"

printf '%s  %s\n' 3e3d04a930ad42e14e97bc68d0e5844eefde702aa3d0a2324e603e5e9b5d0d37 "$checkpoint" | sha256sum -c
printf '%s  %s\n' 454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a "$dataset/annotations/panoptic_val2017.json" | sha256sum -c
{
    echo "Project commit: $revision"
    docker image inspect --format 'Container image: {{.Id}}' "$image"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    docker run --rm --gpus all --ipc=host --network=none \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=1 \
        -e DETECTRON2_DATASETS=/coco -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        --mount "type=bind,source=$dataset,target=/coco,readonly" \
        --mount "type=bind,source=$checkpoint,target=/checkpoint.pth,readonly" \
        --mount "type=bind,source=$root/stage2,target=/stage2,readonly" \
        --mount "type=bind,source=$output,target=/run-output" \
        --entrypoint bash "$image" -ec '
            revision=$(git -c safe.directory="$PWD" rev-parse HEAD)
            echo "MaskDINO commit: $revision"
            [[ "$revision" == 3831d8514a3728535ace8d4ecc7d28044c42dd14 ]]
            PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python -u /stage2/run.py \
                "$1" --size "$2" --batch "$3" --reference "$4"
        ' bash "$mode" "$size" "$batch" "/stage2/runs/$run_id/probe/$size-1/reference.pt"
} 2>&1 | tee "$output/console.log"
