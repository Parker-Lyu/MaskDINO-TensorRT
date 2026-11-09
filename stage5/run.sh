#!/usr/bin/env bash
set -euo pipefail
if (( $# < 5 )); then
    echo "Usage: bash stage5/run.sh DATASET STAGE4_RUNS B1_ENGINE RUN_ID PHASE [OPTIONS...]" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dataset=$(realpath -- "$1")
previous=$(realpath -- "$2")
engine=$(realpath -- "$3")
id=$4
shift 4
[[ "$id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
cd "$root"
[[ -z "$(git status --porcelain)" ]] || { echo "Run from a clean checkout" >&2; exit 1; }
image=$(docker image inspect --format '{{.Id}}' "${MASKDINO_IMAGE:-maskdino:stage1}")
output="$root/stage5/runs/$id"
mkdir -p "$(dirname "$output")"
mkdir "$output"
{
    echo "Project commit: $(git rev-parse HEAD)"
    echo "Container image: $image"
    printf 'Phase:'; printf ' %q' "$@"; echo
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    printf '%s  %s\n' 96facc4e2ee6e2bb61b2a55ae634447a6a019b3c90ef5bb88049143c523cb37a "$engine" | sha256sum -c
    printf '%s  %s\n' 454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a "$dataset/annotations/panoptic_val2017.json" | sha256sum -c
    sha256sum "$engine" "$dataset/annotations/panoptic_val2017.json" stage2/sample.json
    find stage3/runs -path '*/build-b*/network.engine' -exec sha256sum {} +
    docker run --rm --gpus all --pid=host --ipc=host --network=none \
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=1 \
        -e DETECTRON2_DATASETS=/coco -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        --mount "type=bind,source=$dataset,target=/coco,readonly" \
        --mount "type=bind,source=$previous,target=/previous,readonly" \
        --mount "type=bind,source=$engine,target=/b1.engine,readonly" \
        --mount "type=bind,source=$root,target=/project,readonly" \
        --mount "type=bind,source=$output,target=/output" \
        --entrypoint bash "$image" -ec '
            revision=$(git -c safe.directory="$PWD" rev-parse HEAD)
            echo "MaskDINO commit: $revision"
            [[ "$revision" == 3831d8514a3728535ace8d4ecc7d28044c42dd14 ]]
            export PYTHONPATH="/project:$PWD${PYTHONPATH:+:$PYTHONPATH}"
            nvcc --version
            dpkg-query -W libnvinfer10 libjpeg-turbo8
            bash /project/stage5/build.sh /output
            exec python -u -m stage5.suite "$@"
        ' bash "$@"
} 2>&1 | tee "$output/console.log"
