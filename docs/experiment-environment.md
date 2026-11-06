# Experiment environment

## RTX 4090 experiment host

| Component | Recorded configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB reported by `nvidia-smi` |
| CPU | Intel Core i9-14900KF, 24 cores and 32 logical CPUs |
| System memory | About 62 GiB visible to Ubuntu |
| Code and dataset storage | Samsung MZVL22T0HBLB-00B00 NVMe SSD, 1.9 TiB device, ext4 |
| OS and kernel | Ubuntu 24.04.4 LTS, Linux `7.0.0-42-generic` |
| Docker Engine | `29.1.3` |

## COCO 2017 panoptic validation assets

Download the [COCO val2017 images](https://images.cocodataset.org/zips/val2017.zip) and [COCO panoptic annotations](https://s3.amazonaws.com/images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip). Arrange the validation files relative to your dataset root as follows.

| Contents | Path relative to the dataset root |
| --- | --- |
| 5,000 validation JPEGs | `val2017/` |
| Panoptic ground-truth JSON | `annotations/panoptic_val2017.json` |
| 5,000 panoptic ground-truth RGB PNGs | `panoptic_val2017/` |

For container runs, bind-mount the dataset root read-only and pass paths under the mount to the evaluator.

All 5,000 image and mask pairs decoded and matched the JSON filenames and dimensions. All 56,728 segment IDs and pixel areas matched the masks, and all segment categories were among the 133 declared categories. The JSON SHA-256 is `454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a`. These checks cover ground-truth integrity. See the [Stage 1 results](../README.md) for the separate PQ evaluation.

## MaskDINO R50 panoptic checkpoint

- Checkpoint path relative to the project root is `weights/maskdino_r50_panoptic.pth`. The binary is excluded from Git.
- Use the MaskDINO model zoo's COCO panoptic R50 model trained for 50 epochs with 300 queries. The checkpoint is the [official release asset](https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth).
- Checkpoint SHA-256 is `3e3d04a930ad42e14e97bc68d0e5844eefde702aa3d0a2324e603e5e9b5d0d37`.

## Additional Stage 1 annotation

Extract `annotations/instances_val2017.json` from the [COCO train/val annotations](https://images.cocodataset.org/annotations/annotations_trainval2017.zip) and pass its path to the Stage 1 launcher.

Instance validation JSON SHA-256 is `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. Its 5,000 image IDs, filenames, and dimensions match the panoptic validation JSON.

The official evaluation enables the instance, semantic, and panoptic branches.
The script generates semantic labels from the panoptic annotations in the run
output directory.

## Stage 1 reference environment

The values below describe the recorded 2026-11-06 rerun. Check current versions
before reproducing it.

| Item | Recorded value |
| --- | --- |
| NVIDIA driver | `595.84` |
| PyTorch | `2.10.0a0+b558c986e8.nv25.11` |
| PyTorch CUDA version | `13.0` |
| Local Docker image ID | `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e` |
| MaskDINO commit | `3831d8514a3728535ace8d4ecc7d28044c42dd14` |

Use [Stage 1 reproduction instructions](../stage1/README.md) for the command,
[the launcher](../stage1/run.sh) for the exact mounts and checks, and
[the recorded output](../stage1/results/official-r50-20261106-6eb01c1.log) for the measured result.
The image ID identifies the rerun environment. The original run used image ID
`sha256:9758beb33f78b9f983819d835179405de9a7cefd0ea6691726d3bb5a5a8eec4d`. For a fresh run, build the
[Stage 1 image](../maskdino/Dockerfile) as shown in the reproduction instructions.

Before launching, run `nvidia-smi` and `docker image inspect <image>` on the
GPU host. The launcher checks the checkpoint and panoptic JSON hashes. Verify
the instance JSON against the hash above when changing its source or path.
