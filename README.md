# MaskDINO-TensorRT

本项目面向 RTX 4090 台式机上的 MaskDINO R50 全景分割部署，目标是用 TensorRT 优化推理，并在统一口径下评估准确率、延迟、吞吐和显存。Stage 1 先用官方 PyTorch 流程在 COCO 2017 验证集上建立准确率基线。

实验平台为 NVIDIA GeForce RTX 4090（24 GB 显存）、Intel Core i9-14900KF（24 核、32 线程）和 Ubuntu 24.04 LTS。系统可见内存约 62 GiB，代码和数据位于 NVMe SSD。[完整环境](docs/experiment-environment.md)

## Stage 1 · 官方 PyTorch 基线

在 RTX 4090 上使用 MaskDINO 官方 R50 panoptic 配置和权重评估 COCO 2017 val 的 5,000 张图片。模型和前后处理走官方 `train_net.py`，PQ 由 Detectron2 的 `COCOPanopticEvaluator` 调用 COCO panopticapi 计算。

| 来源 | PQ | PQ Things | PQ Stuff |
| --- | ---: | ---: | ---: |
| [RTX 4090 实测](stage1/results/official-r50-20261106-6eb01c1.log) | **52.9567** | 58.9645 | 43.8884 |
| [MaskDINO 官方 Model Zoo](https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/README.md#coco-panoptic-segmentation) | 53.0 | — | — |

实测 PQ 按官方表格的一位小数精度为 **53.0**，预测覆盖全部 5,000 个真值图像 ID。

[Stage 1 复现步骤](stage1/README.md) · [验收 tag `stage1-official-pytorch`](https://github.com/Parker-Lyu/MaskDINO-TensorRT/tree/stage1-official-pytorch) · [环境、数据与权重](docs/experiment-environment.md)

## Stage 2 · 方形输入 PyTorch 评估

在 RTX 4090 上以 FP32、B4 分别评估三档方形输入，每档覆盖完整的 COCO 2017 val。

| 输入 | PQ | PQ Things | PQ Stuff | 相对 Stage 1 ΔPQ |
| --- | ---: | ---: | ---: | ---: |
| 640×640 | 49.9662 | 54.8245 | 42.6330 | −2.9905 |
| 800×800 | 52.0120 | 57.3157 | 44.0065 | −0.9447 |
| 1024×1024 | **53.1862** | 59.0272 | 44.3697 | +0.2295 |

1024×1024 作为 TensorRT 优化主线。PyTorch B1 的离线端到端 P95 为 51.93 ms，吞吐为 20.38 图/s。B8 的吞吐增至 22.56 图/s，P95 为 359.10 ms。性能数据使用同一份 500 张清单，图中为三轮中位数。

![Stage 1 与 Stage 2 的 COCO PQ，以及 1024 方形输入的离线吞吐与延迟](stage2/results/comparison.svg)

[Stage 2 复现与结果](stage2/README.md) · [固定 500 张性能样本](stage2/sample.json) · 验收 tag `stage2-pytorch-square`

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| [maskdino/](maskdino/) | 固定上游源码版本、PyTorch 环境和 CUDA 扩展兼容补丁 |
| [stage1/](stage1/) | 官方评估入口、复现命令和结果 |
| [stage2/](stage2/) | 方形 PyTorch 推理、全量 PQ 和性能样本 |
| [docs/](docs/) | 环境、数据及权重说明 |

构建镜像时会检出指定版本的官方 MaskDINO 源码。

## 许可证

[Apache-2.0](LICENSE) · [第三方组件与许可](maskdino/THIRD_PARTY_NOTICES.md)
