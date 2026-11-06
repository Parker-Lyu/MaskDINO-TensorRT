# MaskDINO-TensorRT

本项目面向 RTX 4090 台式机上的 MaskDINO R50 全景分割部署，目标是用 TensorRT 优化推理，并在统一口径下评估准确率、延迟、吞吐和显存。

实验平台为 NVIDIA GeForce RTX 4090（24 GB 显存）、Intel Core i9-14900KF（24 核、32 线程）和 Ubuntu 24.04 LTS。系统可见内存约 62 GiB，代码和数据位于 NVMe SSD。[完整环境](docs/experiment-environment.md)

## Stage 1 · 官方 PyTorch 基线

在 RTX 4090 上使用 MaskDINO 官方 R50 panoptic 配置和权重评估 COCO 2017 val 的 5,000 张图片。模型和前后处理走官方 `train_net.py`，PQ 由 Detectron2 的 `COCOPanopticEvaluator` 调用 COCO panopticapi 计算。

| 来源 | PQ | PQ Things | PQ Stuff |
| --- | ---: | ---: | ---: |
| [RTX 4090 实测](stage1/results/official-r50-20261106-6eb01c1.log) | **52.9567** | 58.9645 | 43.8884 |
| [MaskDINO 官方 Model Zoo](https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/README.md#coco-panoptic-segmentation) | 53.0 | — | — |

实测 PQ 按官方表格的一位小数精度为 **53.0**，预测覆盖全部 5,000 个真值图像 ID。

[Stage 1 复现步骤](stage1/README.md) · [验收 tag `stage1-official-pytorch`](https://github.com/Parker-Lyu/MaskDINO-TensorRT/tree/stage1-official-pytorch) · [环境、数据与权重](docs/experiment-environment.md)

## Stage 2 · 方形输入与 PyTorch 性能基线

本阶段为 TensorRT 建立可复用的输入规格和 PyTorch benchmark。按以下顺序完成五项工作。

1. **统一输入形状。** 等比缩放到指定长边，右侧和底部填充到正方形，让同一尺寸的图片可以直接组成 batch。后处理裁掉填充区域，再恢复到原图坐标。
2. **固定两套样本。** 32 张用于验证预处理几何和跨 batch 的数值及分割输出，500 张用于重复测量性能。两套 ID 均保存在[样本清单](stage2/sample.json)。
3. **比较三档准确率。** 640、800、1024 均用 FP32、B4 跑完整 COCO 2017 val。1024 的 PQ 最高，800 作为速度对照。
4. **对比 800 与 1024 的性能。** 两档均测 B1 和 B4。1024 比 800 高 **1.1742 PQ**，B1 的端到端 P95 多 **14.08 ms**。为优先保留准确率，选定 **1024×1024** 作为后续基线尺寸。
5. **补齐 1024 的 batch 数据。** 增测 B2 和 B8。B8 相比 B1 的离线吞吐提高 **10.7%**，P95 从 **51.93 ms** 增到 **365.04 ms**，因此 TensorRT 优先优化 **1024/B1**。

### 全量 COCO PQ

| 输入 | Batch | PQ | PQ Things | PQ Stuff | 相对 Stage 1 ΔPQ | 覆盖图数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 官方输入 | 1 | 52.9567 | 58.9645 | 43.8884 | — | 5,000 |
| 640×640 | 4 | 49.9662 | 54.8245 | 42.6330 | −2.9905 | 5,000 |
| 800×800 | 4 | 52.0120 | 57.3157 | 44.0065 | −0.9447 | 5,000 |
| 1024×1024 | 4 | **53.1862** | 59.0272 | 44.3697 | +0.2295 | 5,000 |

### RTX 4090 性能对照

每组使用同一份 500 张清单、FP32，每轮预热 20 个 batch，共测三轮。表中为三轮中位数。吞吐按整轮完成图数除以墙钟时间，延迟和吞吐均使用离线负载。

| 输入 | Batch | 网络 GPU P50 / P95（ms/批） | 端到端 P50 / P95（ms/批） | 离线端到端吞吐（图/s） | 峰值 allocated / reserved（GiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 800×800 | 1 | 26.49 / 26.77 | 35.02 / 37.85 | 28.35 | 2.07 / 3.92 |
| 800×800 | 4 | 87.61 / 88.14 | 122.15 / 127.34 | 32.72 | 2.37 / 4.83 |
| **1024×1024** | **1** | 37.77 / 37.88 | 48.79 / 51.93 | 20.38 | 2.58 / 4.71 |
| 1024×1024 | 2 | 71.22 / 71.44 | 93.47 / 97.81 | 21.33 | 2.74 / 5.59 |
| 1024×1024 | 4 | 138.83 / 139.37 | 183.56 / 189.70 | 21.82 | 3.06 / 4.91 |
| 1024×1024 | 8 | 264.16 / 274.32 | 355.51 / 365.04 | 22.56 | 5.06 / 7.25 |

![Stage 1 与 Stage 2 的 COCO PQ，以及 1024 方形输入的离线吞吐与批次端到端 P95](stage2/results/comparison.svg)

表格和右图均按批统计端到端延迟，从该批开始处理内存 JPEG，计到最后一张图的 CPU 输出就绪。

[Stage 2 复现步骤、运行版本与原始结果](stage2/README.md) · 验收 tag `stage2-pytorch-square`

## Stage 3 · TensorRT 单请求后端

固定 1024×1024、batch=1、单 slot，两种后端共用 Stage 2 的预处理和 panoptic 后处理。PyTorch FP32 与 TensorRT 均关闭 TF32。每种配置评估 5,000 个唯一预测 ID，性能使用同一 500 张清单，每轮预热 20 次，表中为三轮中位数。

| 网络后端 | PQ / PQ_th / PQ_st | 网络 GPU P50 / P95（ms） | 端到端 P50 / P95（ms） | 吞吐（图/s） | 整卡采样峰值（GiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 53.1863 / 59.0273 / 44.3696 | 37.58 / 37.75 | 48.68 / 51.87 | 20.44 | 5.79 |
| TensorRT FP32 | 53.1863 / 59.0273 / 44.3696 | 20.65 / 21.79 | 31.91 / 35.18 | 31.18 | 5.59 |
| TensorRT，最终 mask 投影 FP16 | 53.1882 / 59.0305 / 44.3697 | 20.82 / 21.56 | 31.92 / 35.16 | 31.10 | 5.59 |

TensorRT FP32 使网络 P95 降低 **42.3%**、端到端 P95 降低 **32.2%**、吞吐提高 **52.5%**。最终混合版通过逐图和全量准确率验收，端到端 P95 中位数比 TensorRT FP32 低 0.024 ms，落在两组三轮的波动范围内。

![Stage 1–3 的 COCO PQ，以及 Stage 3 的同口径 1024 B1 性能](stage3/results/comparison.svg)

[Stage 3 复现步骤、原始结果与 Nsight 分析](stage3/README.md) · 验收 tag `stage3-tensorrt-b1`

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| [maskdino/](maskdino/) | 固定上游源码版本、PyTorch 环境和 CUDA 扩展兼容补丁 |
| [stage1/](stage1/) | 官方评估入口、复现命令和结果 |
| [stage2/](stage2/) | 方形 PyTorch 推理、全量 PQ 和性能样本 |
| [stage3/](stage3/) | 单 slot TensorRT 网络、FP32 与混合精度验收、Nsight 基线 |
| [docs/](docs/) | 环境、数据及权重说明 |

构建镜像时会检出指定版本的官方 MaskDINO 源码。

## 许可证

[Apache-2.0](LICENSE) · [第三方组件与许可](maskdino/THIRD_PARTY_NOTICES.md)
