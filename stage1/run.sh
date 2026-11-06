#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "Usage: bash stage1/run.sh PANOPTIC_DATASET_ROOT INSTANCES_VAL_JSON [RUN_ID]" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dataset=$(realpath -- "$1")
instances=$(realpath -- "$2")
run_id=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
image=${MASKDINO_IMAGE:-maskdino:stage1}
checkpoint="$root/weights/maskdino_r50_panoptic.pth"
printf '%s  %s\n' 3e3d04a930ad42e14e97bc68d0e5844eefde702aa3d0a2324e603e5e9b5d0d37 "$checkpoint" | sha256sum -c
printf '%s  %s\n' 454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a "$dataset/annotations/panoptic_val2017.json" | sha256sum -c
[[ -d "$dataset/val2017" && -d "$dataset/panoptic_val2017" && -f "$instances" ]]
mkdir -p "$root/stage1/runs"
output="$root/stage1/runs/$run_id"
mkdir "$output"

docker run --rm -i --gpus all --ipc=host --network=none \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=1 \
    -e DETECTRON2_DATASETS=/run-output/datasets -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    --mount "type=bind,source=$dataset,target=/coco-original,readonly" \
    --mount "type=bind,source=$instances,target=/instances_val2017.json,readonly" \
    --mount "type=bind,source=$checkpoint,target=/checkpoint.pth,readonly" \
    --mount "type=bind,source=$output,target=/run-output" \
    --entrypoint bash "$image" -s 2>&1 <<'CONTAINER' | tee "$output/console.log"
set -euo pipefail
revision=$(git -c safe.directory="$PWD" rev-parse HEAD)
echo "MaskDINO commit: $revision"
[[ "$revision" == 3831d8514a3728535ace8d4ecc7d28044c42dd14 ]]
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)'

coco="$DETECTRON2_DATASETS/coco"
mkdir -p "$coco/annotations"
ln -s /coco-original/val2017 /coco-original/panoptic_val2017 "$coco/"
ln -s /coco-original/annotations/panoptic_val2017.json /instances_val2017.json "$coco/annotations/"
PYTHONPATH="$PWD/datasets${PYTHONPATH:+:$PYTHONPATH}" python -u - <<'PYTHON'
import os
from pathlib import Path

from prepare_coco_semantic_annos_from_panoptic_annos import (
    COCO_CATEGORIES,
    separate_coco_semantic_from_panoptic,
)

coco = Path(os.environ["DETECTRON2_DATASETS"]) / "coco"
separate_coco_semantic_from_panoptic(
    str(coco / "annotations/panoptic_val2017.json"),
    str(coco / "panoptic_val2017"),
    str(coco / "panoptic_semseg_val2017"),
    COCO_CATEGORIES,
)
PYTHON

python -u train_net.py --eval-only --num-gpus 1 \
    --config-file configs/coco/panoptic-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml \
    MODEL.WEIGHTS /checkpoint.pth OUTPUT_DIR /run-output/evaluation
CONTAINER
