# 实验环境

[English](experiment-environment.md) · **简体中文**

## RTX 4090 实验主机

| 组件 | 记录配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，`nvidia-smi` 报告 24,564 MiB |
| CPU | Intel Core i9-14900KF，24 核、32 个逻辑 CPU |
| 系统内存 | Ubuntu 可见约 62 GiB |
| 代码和数据集存储 | Samsung MZVL22T0HBLB-00B00 NVMe SSD，设备容量 1.9 TiB，ext4 |
| 操作系统和内核 | Ubuntu 24.04.4 LTS，Linux `7.0.0-42-generic` |
| Docker Engine | `29.1.3` |

## COCO 2017 全景分割验证数据

下载 [COCO val2017 图像](https://images.cocodataset.org/zips/val2017.zip)和 [COCO panoptic 标注](https://s3.amazonaws.com/images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)。按以下相对路径在数据集根目录下放置验证文件。

| 内容 | 相对数据集根目录的路径 |
| --- | --- |
| 5,000 张验证集 JPEG | `val2017/` |
| 全景分割真值 JSON | `annotations/panoptic_val2017.json` |
| 5,000 张全景分割真值 RGB PNG | `panoptic_val2017/` |

在容器中运行时，将数据集根目录以只读方式绑定挂载，并将挂载目录下的路径传给评估器。

5,000 对图像和 mask 均可解码，且文件名和尺寸均与 JSON 匹配。全部 56,728 个 segment ID 和像素面积均与 mask 匹配，segment 类别均属于声明的 133 个类别。JSON 的 SHA-256 为 `454873a8a01114246066ac841750eb742df3b5e42ce927ef38b49690084ec75a`。这些检查验证真值数据的完整性。独立的 PQ 评估见 [Stage 1 结果](../README.zh-CN.md)。

## MaskDINO R50 全景分割权重

- 权重相对项目根目录的路径为 `weights/maskdino_r50_panoptic.pth`。该二进制文件不纳入 Git。
- 使用 MaskDINO Model Zoo 中训练 50 个 epoch、采用 300 个 query 的 COCO panoptic R50 模型。权重来自[官方发布资产](https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth)。
- 权重 SHA-256 为 `3e3d04a930ad42e14e97bc68d0e5844eefde702aa3d0a2324e603e5e9b5d0d37`。

## Stage 1 的额外标注

从 [COCO train/val 标注](https://images.cocodataset.org/zips/annotations_trainval2017.zip)解压 `annotations/instances_val2017.json`，并将路径传给 Stage 1 启动脚本。

实例验证集 JSON 的 SHA-256 为 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。其中 5,000 个图像 ID、文件名及尺寸均与 panoptic 验证集 JSON 一致。

官方评估启用实例、语义和全景分割分支。脚本从 panoptic 标注生成语义标签，并写入本次运行的输出目录。

## Stage 1 参考环境

以下为 2026-11-06 重跑的记录。复现前请核对当前版本。

| 项目 | 记录值 |
| --- | --- |
| NVIDIA 驱动 | `595.84` |
| PyTorch | `2.10.0a0+b558c986e8.nv25.11` |
| PyTorch CUDA 版本 | `13.0` |
| 本地 Docker 镜像 ID | `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e` |
| MaskDINO 提交 | `3831d8514a3728535ace8d4ecc7d28044c42dd14` |

运行命令见 [Stage 1 复现步骤](../stage1/README.zh-CN.md)，具体挂载和校验见[启动脚本](../stage1/run.sh)，实测结果见[运行记录](../stage1/results/official-r50-20261106-6eb01c1.log)。
表中的镜像 ID 对应重跑环境。原始运行的镜像 ID 为 `sha256:9758beb33f78b9f983819d835179405de9a7cefd0ea6691726d3bb5a5a8eec4d`。重新运行时，按复现步骤构建 [Stage 1 镜像](../maskdino/Dockerfile)。

启动前，在 GPU 主机上执行 `nvidia-smi` 和 `docker image inspect <image>`。
启动脚本会校验权重与 panoptic JSON 的哈希。更换实例标注 JSON 的来源或路径时，请与上述哈希核对。
