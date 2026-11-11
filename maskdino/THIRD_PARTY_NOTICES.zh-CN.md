# 第三方来源

[English](THIRD_PARTY_NOTICES.md) · **简体中文**

| 组件 | 来源 | 许可证或条款 |
| --- | --- | --- |
| MaskDINO | 固定版本的官方仓库 | Apache-2.0 |
| Detectron2 | 固定版本的官方仓库 | Apache-2.0 |
| panopticapi | 固定版本的官方仓库 | BSD-2-Clause |
| NVIDIA PyTorch 容器 | NGC 25.11，锁文件记录了参考镜像 ID | NVIDIA 容器条款 |
| R50 权重 | MaskDINO 官方 Model Zoo | 从原始来源单独下载 |
| COCO 2017 验证数据 | COCO 数据集 | 按原条款单独下载 |

来源 URL、固定版本及权重哈希记录在 [dependencies.lock.json](dependencies.lock.json) 中。获取的上游源码树保留原始声明。各固定版本的许可证副本见 [licenses/](licenses/)。

[兼容补丁](patches/maskdino-pytorch.patch)更新了上游 CUDA 扩展中已弃用的 PyTorch C++ API 调用，保留了算子算法。

Feng Li 等人，*Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation*，CVPR 2023。[论文](https://arxiv.org/abs/2206.02777) · [官方源码](https://github.com/IDEA-Research/MaskDINO)。
