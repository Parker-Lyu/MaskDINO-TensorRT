# Third-party sources

| Component | Source | License or terms |
| --- | --- | --- |
| MaskDINO | Pinned official repository revision | Apache-2.0 |
| Detectron2 | Pinned official repository revision | Apache-2.0 |
| panopticapi | Pinned official repository revision | BSD-2-Clause |
| NVIDIA PyTorch container | NGC 25.11, reference image ID recorded in the lock file | NVIDIA container terms |
| R50 checkpoint | Official MaskDINO model zoo | Downloaded separately from the original source |
| COCO 2017 validation data | COCO dataset | Downloaded separately under the original terms |

Source URLs, revisions and the checkpoint hash are in [dependencies.lock.json](dependencies.lock.json). Upstream source notices are preserved in the fetched trees. License copies for the pinned revisions are in [licenses/](licenses/).

The [compatibility patch](patches/maskdino-pytorch.patch) updates deprecated PyTorch C++ API calls in the upstream CUDA extension and preserves the operator algorithm.

Feng Li et al., *Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation*, CVPR 2023. [Paper](https://arxiv.org/abs/2206.02777) · [Official source](https://github.com/IDEA-Research/MaskDINO).
