# Stage 6 · TensorRT 逐层 profiling 与局部混合精度

在 RTX 4090 上固定 MaskDINO R50 panoptic、1024×1024、B1、模型级 FP16 autocast / TF32 on 和 Stage 4–5 的 C++/CUDA 流水线。先定位网络热点，再分别验证 FPN 单层卷积和 encoder 六层 FFN。FPN 候选通过本阶段的离线与精度验收。六层 FFN 候选提供性能和 PQ 对照，逐图为 26/32，保留为实验配置。

| 实验 | 问题 | 主要结果 | 记录 |
| --- | --- | --- | --- |
| 逐层 profiling | 找出耗时最多的单层与重复算子 | FPN 3×3 卷积为最贵单层，0.903 ms。Encoder FFN 的 12 个 GEMM 合计 3.675 ms | [采集、归属与计数器](layer-profile.md) |
| FPN 单层 FP16 | 只改变通向 masks 的卷积，验证 Tensor Core 收益 | B1/S1 端到端 P95 降 2.6%，B1/S2 稳态吞吐增 2.9%。两组均 32/32，全量 PQ 53.1718 | [对照与验收](fpn-half-ab.md) |
| Encoder 六层 FFN FP16 | 在已验收 FPN engine 上同时改 12 个 GEMM | B1/S1 端到端 P95 降 13.7%，B1/S2 稳态吞吐增 16.8%。全量 PQ 53.1856，逐图 26/32 | [性能、失败样本与 PQ](ffn-full-half.md) |

## 基线和测量边界

- **基础 engine。** 从已验收的混合精度 B1 plan 开始，SHA-256 为 `ab6054100b6256861ad320df440543d2e82cf2a833db2e77c159b736021401cf`。FPN 改动通过后，以其 plan `d80102c6…` 作为 FFN 对照。权重、COCO 样本、前后处理、TensorRT FP32 I/O 和 B1 形状沿用原流水线。[原始构建](../stage3/results/precision/build-autocast-on/build.json) · [FPN 构建](results/fpn-half-ab/cached/fpn-half/build.json)
- **准确率。** 32 张先与对应的原生 B1 对照逐图比较，门槛为类别一致率 ≥99.9%、全部 segment 配对、同类 IoU ≥0.98。全量使用 COCO val2017 的 5,000 张、验证唯一预测 ID，并以同输入 PyTorch FP32 为基线，PQ、PQ Things 和 PQ Stuff 各允许下降最多 0.05 点。[输入样本](../stage2/sample.json) · [PyTorch 基线](../stage3/results/pt-pq/result.json)
- **正式离线计时。** 同一份 500 张内存 JPEG，每个 slot 预热 20 批，三轮轮换 plan 顺序。网络 GPU P50/P95 用 CUDA events，端到端 P50/P95 从派发计到原图尺寸的 CPU map 和 metadata 就绪。稳态吞吐使用墙钟中间 90% 的窗口。NVML 每 20 ms 采样推理进程显存。正式计时与逐层 profiling、Nsight 分开执行。[FPN 逐轮结果](results/fpn-half-ab/cached/ab/summary.json) · [FFN 逐轮结果](results/ffn-full-half/ab/summary.json)
- **环境。** RTX 4090、驱动 595.84、CUDA Toolkit 13.0.88、TensorRT 10.14.1.48、参考容器 `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e`。GPU 空闲检查、进程采样和输入哈希随各轮日志归档。[实验资产](../docs/experiment-environment.md) · [逐层实验日志](results/layer-profile-b1/console.log)

## 定位热点

固定 B1/S1，用同一张原生预处理后的 COCO 图跑三轮，再换两张图复查排序。通过 `trtexec --separateProfileRun --dumpProfile` 采集层均值，按层或重复算子组汇总，占比分母为逐层计时均值之和。另用 Nsight Systems 将 NVTX 层名和 CUDA launch 对应到实际 kernel，选代表 kernel 用 Nsight Compute 查看资源计数器。

| 层或组 | 逐层耗时（ms/图） | 占逐层时间 |
| --- | ---: | ---: |
| Encoder FFN 的 12 个 GEMM | 3.675 | 24.30% |
| Encoder MSDeformAttn 的 6 个插件层 | 1.480 | 9.78% |
| FPN `/sem_seg_head/layer_1/Conv` | 0.903 | 5.97% |
| `input_proj.3` 的 3×3 卷积 | 0.306 | 2.02% |
| 最终 mask Einsum 和 FP32 输出转换 | 0.171 | 1.13% |

FPN 卷积采用 FP32 Winograd，代表 kernel 的 Tensor pipe 活跃为 0%、L2 吞吐约占峰值的 74.3%。代表性 FFN `linear2` 已使用 TF32 Tensor Core，测得 DRAM 吞吐约峰值的 67.8%、活跃 warp 为 14.6%。这些计数器用于提出优化假设。选择先做 FPN，是因为它是最贵单层且 ONNX 数据流只通向 `masks`，而 FFN 位于 proposal TopK 上游。[逐层汇总](results/layer-profile-b1/summary.json) · [Nsight Systems 原始报告](results/layer-profile-b1/layer-kernels.nsys-rep) · [ONNX 数据流](results/layer-profile-b1/onnx-paths.txt)

## FPN 单层 FP16

在 `/sem_seg_head/layer_1/Conv` 将输入和权重转为 FP16，卷积输出在 GroupNorm 前转回 FP32。首轮独立构建的候选为 31/32。对异常图 `470924` 交叉替换 logits 和 masks，确认缺失 segment 随候选 logits 出现。随后用共享 timing cache 构建对照和候选。两份 engine 的 60 个同名卷积或池化层中，只有目标卷积更换 tactic。异常图的 logits 和 boxes 在这对 engine 中逐元素一致，候选 S1 和 S2 都通过 32/32。[首轮张量对照](results/fpn-half-ab/profile/470924-probe.json) · [共享 cache 复测](results/fpn-half-ab/cached/ab/470924-probe.json)

| 指标 | 共享 cache 对照 | FPN 单层 FP16 |
| --- | ---: | ---: |
| 目标卷积均值（ms） | 0.904 | 0.251 |
| 相邻 Add 至 GroupNorm/ReLU 合计（ms） | 1.279 | 0.820 |
| B1/S1 端到端 P95（ms） | 16.27 | 15.85 |
| B1/S2 稳态吞吐（图/s） | 74.21 | 76.33 |

S1、S2 各完成 5,000 张，PQ / PQ Things / PQ Stuff 均为 **53.17180 / 59.01222 / 44.35606**，通过全量门槛。FPN 验收提交为 `5f7b57f`。验收范围是逐图、全量 PQ 和 500 张离线配对。 [逐图与全量结果](fpn-half-ab.md#全量-coco-pq)

![FPN 单层优化的离线对照](results/fpn-half-ab/comparison.svg)

## Encoder 六层 FFN FP16

在 FPN 已验收 engine 上，将 encoder 第 0–5 层的 `linear1 → bias → ReLU → linear2 → bias` 保持 FP16，在残差加法和 LayerNorm 前转回 FP32。导入 FPN 构建留下的 timing cache。Inspector 确认 12 个 GEMM 均使用 FP16 Tensor Core，六个 `linear2` 融合了转换和残差/归一化的相关计算。[候选构建](results/ffn-full-half/build/build.json) · [Engine Inspector](results/ffn-full-half/build/layers.json)

| 指标 | 同轮 FPN 对照 | 六层 FFN FP16 |
| --- | ---: | ---: |
| B1/S1 网络 GPU P95（ms） | 14.01 | 11.78 |
| B1/S1 端到端 P95（ms） | 16.10 | 13.90 |
| B1/S1 稳态吞吐（图/s） | 64.64 | 75.41 |
| B1/S2 端到端 P95（ms） | 27.87 | 24.28 |
| B1/S2 稳态吞吐（图/s） | 76.39 | 89.22 |

两种 slot 各完成 5,000 张，PQ / PQ Things / PQ Stuff 均为 **53.18563 / 59.02411 / 44.37283**，相对 PyTorch 的差值为 **−0.00062 / −0.00315 / +0.00319** 点。两种 slot 的 32 张输出逐字节一致，均只有 **26/32** 通过。最差样本 `435081` 的类别一致率为 **85.68%**，最低同类 IoU 为 **0.166**。逐图结果决定该版仍是实验配置。[六张失败图与全量 PQ](ffn-full-half.md) · [PQ 与性能汇总](results/ffn-full-half/summary.json)

![六层 FFN FP16 的离线对照](results/ffn-full-half/comparison.svg)

## 复现和归档

从干净 checkout 开始，确认 GPU 空闲，并核对参考镜像、原始 ONNX、engine 和数据的 SHA-256。

| 实验 | 原始运行提交 | 本分支复现入口 | 归档 |
| --- | --- | --- | --- |
| 逐层计时与 Nsight | `8e3964b` | [采集脚本及 Nsight 命令](layer-profile.md#复现) | [层耗时、时间线和计数器](results/layer-profile-b1/) |
| FPN 首轮构建、计时与张量交叉验证 | `672f0aa`、`d9e30d7`、`9237433` | [ONNX 改写与原生 A/B](fpn-half-ab.md#复现入口) | [无共享 cache 结果](results/fpn-half-ab/alignment/) |
| FPN 共享 cache、原生 A/B 和全量 PQ | `54e6f15` | [共享 cache 构建、计时和 PQ 命令](fpn-half-ab.md#复现入口) | [配对构建与计时](results/fpn-half-ab/cached/) · [全量 PQ](results/fpn-half-ab/pq/) |
| 六层 FFN FP16 | `301764c` | [ONNX 改写、计时和 PQ 命令](ffn-full-half.md#复现入口) | [原生 A/B、逐层计时与 PQ](results/ffn-full-half/) |

本分支的复现脚本使用 `stage6/` 入口和 `stage6/runs/` 实验输出目录。检出原始运行提交复跑时使用该提交里的脚本路径。4090 上的历史输出路径见各详细报告。此前 batch × slot 和在线负载实验仍可从历史提交 `fcc7dc9` 的 `stage6/` 读取。
