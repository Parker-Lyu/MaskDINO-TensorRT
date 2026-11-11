# Stage 1 · Reproduce Original Model Accuracy

**English** · [简体中文](README.zh-CN.md)

Use official MaskDINO commit `3831d8514a3728535ace8d4ecc7d28044c42dd14`, the 50-epoch R50 panoptic configuration, and its checkpoint. Resize the input short side to 800 and cap the long side at 1333. Keep the official configuration's panoptic, instance, and semantic output branches.

## Reproduction

On a Linux x86_64 GPU host, install Docker, NVIDIA Container Toolkit, and a driver compatible with CUDA 13. The current build targets SM 8.9 on the RTX 4090. Prepare the COCO files using the [data and weights instructions](../docs/experiment-environment.md). The download script fetches the checkpoint and verifies its hash.

Run from the repository root.

```bash
python3 maskdino/scripts/fetch.py assets
docker build --network=host -f maskdino/Dockerfile -t maskdino:stage1 .
bash stage1/run.sh \
  /path/to/coco2017-panoptic-val \
  /path/to/instances_val2017.json
```

The [Dockerfile](../maskdino/Dockerfile) specifies the base image tag, pins dependencies, and applies the [CUDA extension compatibility patch](../maskdino/patches/maskdino-pytorch.patch). The [dependency lock file](../maskdino/dependencies.lock.json) records the reference base image ID. The launcher defaults to `maskdino:stage1`. Set `MASKDINO_IMAGE` to use another image built from the same recipe.

The launcher verifies the SHA-256 hashes of the checkpoint and panoptic annotation JSON, mounts inputs read-only, calls the official preprocessing script to generate semantic labels, and runs the official evaluator. Logs and predictions go to `stage1/runs/<run-id>/`.

## Results

The official evaluation was rerun on an RTX 4090 on 2026-11-06 from project commit `6eb01c17745e10756558464f3f85b35e76bbca09`. Predictions cover all 5,000 COCO 2017 validation images. PQ was **52.9567**, PQ Things was **58.9645**, and PQ Stuff was **43.8884**, matching the original evaluation.

[Rerun log excerpt and coverage check](results/official-r50-20261106-6eb01c1.log) · [Original evaluation log excerpt](results/official-r50-20261106.log)
