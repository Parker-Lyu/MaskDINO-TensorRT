# Stage 3 · TensorRT 单请求网络后端

固定 MaskDINO R50 panoptic、1024×1024、batch=1、单 slot。前处理、归一化、panoptic 后处理和 COCO evaluator 共用 Stage 2 实现。PyTorch FP32 与 TensorRT 均关闭 TF32。

## 验收标准

- 复用 Stage 2 的 32 张验证清单。逐图报告 logits、masks、boxes 的最大绝对误差、平均绝对误差和相对 L2。张量形状须一致且所有值有限。当 mask 的直接相对 L2 大于 0.02 时，使用下采样 mask 余弦距离与类别概率距离做匈牙利匹配，附上配对后误差。匹配只用于诊断，后处理仍使用原始输出。
- 类别图比较包含 void，并区分 thing 和 stuff。对同类 segment 按 IoU 从高到低一对一配对，记录原始 metadata 是否相同、每个匹配的 IoU 和未匹配数量。FP32 逐图类别一致率至少 99.9%，所有 segment 均配对且 IoU 至少 0.98。混合精度分别为 99% 和 0.90。异常保存原始输入、输出张量、panoptic map、metadata 和差异图。
- 全量 5,000 张预测 ID 唯一且与真值集合一致。FP32 相对 PyTorch B1 的 PQ、PQ_th、PQ_st 各项下降最多 0.05 点，混合精度最多 0.20 点。Stage 2 B4 的 PQ 53.1862 单独列作参考。
- 每种后端用相同顺序跑 Stage 2 的 500 张性能清单，每轮预热 20 次，共三轮。内存 JPEG 到 CPU panoptic map 与 metadata 就绪为端到端边界。GPU 网络时间用 CUDA events。整卡 NVML 20 ms 采样涵盖 TensorRT 内存及桌面占用，记录观测峰值。
- Nsight Systems 单独采集，NVTX 覆盖解码、前处理、H2D、GPU 归一化、网络、后处理、D2H。

## 复现入口

在 4090 上从干净的提交运行。准备 Stage 2 镜像及数据，设置 `MASKDINO_CHECKPOINT` 为权重路径。每次运行用新的 ID。日志和产物位于 `stage3/runs/<ID>/`。容器内可通过 `/project/stage3/runs/<ID>/` 读取之前的产物。
