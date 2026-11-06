# Stage 4 · C++/CUDA 单请求流水线

复用 Stage 3 的 MaskDINO R50 panoptic TensorRT FP32 engine。固定 1024×1024、batch 1、单 slot、单条显式 CUDA stream，TF32 保持关闭。原生入口 `maskdino-native` 直接读取内存 JPEG，输出 CPU int32 panoptic map 和 segment metadata。运行时链接 TensorRT、CUDA、libjpeg、NVML 与系统库。Python 用于组织实验、参考链路、对齐和 COCO 评估。

## 实现与验收

1. CPU 用 libjpeg 解码 RGB，并按 EXIF 方向变换。CUDA 以与 Pillow 一致的像素中心、双线性滤波范围和两次 uint8 取整做 resize，再向右、下填充 128，归一化为 NCHW FP32。保持 Stage 2 的长边尺寸和 ties-to-even 取整。32 张 COCO 与 14 张合成 JPEG 的 RGB、有效尺寸和输入张量逐元素一致。合成 JPEG 覆盖方向 1–8、缩小、灰度与 CMYK。 [逐图数据](results/native-check/preprocess.jsonl)
2. CUDA 按 `align_corners=False` 恢复 masks。对每个原图像素先确定它在裁剪后 1024 网格上的双线性邻点，再从 256×256 mask 计算这些网格值。这样保留两次插值及中间裁剪的运算顺序。7 组 CUDA 与 PyTorch 合成对照逐元素一致。 [插值检查](results/native-check/restore.jsonl)
3. CUDA 按 `sigmoid(logit) > 0.25` 筛选 query，以温度 0.06 的 softmax 概率乘 mask 概率做像素分配。winner 同分时保留最早的 query。面积阈值分别采用 `mask probability >= 0.5` 和 `assigned / original >= 0.8`，按 query 顺序编号并合并同类 stuff。最后一次 stream 同步等待 map 与 metadata 回到 CPU。
4. 原生全链路和同张量后处理对照分别使用 Stage 2 的 32 张清单。两种对照均要求每张类别一致率 ≥99.9%、全部 segment 配对且同类 IoU ≥0.98。首版有 4 张失败，同张量对照复现了差异。Stage 3 使用 `SEMANTIC_CE_LOSS=false`，133 个输出列均为有效类别，CUDA 首版误过滤了最后一个有效类别 132。修正后，两种对照均为 32/32，逐图类别一致率和所有 segment IoU 均为 100%，metadata 全部一致。 [首版](results/native-align/result.json) · [同张量定位](results/isolate/result.json) · [修正后](results/native-align-fixed/result.json)
5. 另用 7 组构造样例逐元素检查空候选、分类阈值等于 0.25、mask 阈值等于 0.5、类别 132、像素同分、覆盖率等于与低于 0.8，以及同类 stuff 合并。用 `9fe64e5` 重跑全部 32 张对齐及这组检查，全部通过。 [边界数据与重跑结果](results/native-boundaries-fixed/result.json)
6. 使用完整 COCO 2017 panoptic val。要求 5,000 个预测 ID 唯一并与真值集合一致，PQ、PQ_th、PQ_st 各项相对 Stage 3 FP32 下降不超过 0.05 点。

## 复现

使用 [Stage 3 参考环境](../stage3/results/environment.json)。正式实验在 RTX 4090 上运行。准备同一镜像、checkpoint、数据目录与 Stage 3 FP32 engine。脚本校验输入资产，要求干净 Git checkout，并将运行提交、镜像 ID、命令与校验摘要写入 `stage4/runs/<ID>/console.log`。各运行 ID 只能使用一次。

```bash
export MASKDINO_IMAGE=maskdino:stage1
export MASKDINO_CHECKPOINT=/path/to/maskdino_r50_panoptic.pth
data=/path/to/coco2017-panoptic-val
engine=/path/to/stage3/runs/build-fp32/network.engine

bash stage4/run.sh "$data" "$engine" native-check native check
bash stage4/run.sh "$data" "$engine" native-align-fixed native align
for round in 1 2 3; do
  case "$round" in
    1) order=(pytorch reference native) ;;
    2) order=(reference native pytorch) ;;
    3) order=(native pytorch reference) ;;
  esac
  for backend in "${order[@]}"; do
    bash stage4/run.sh "$data" "$engine" "perf-$backend-r$round" "$backend" benchmark
  done
done
bash stage4/run.sh "$data" "$engine" trace-reference reference trace
bash stage4/run.sh "$data" "$engine" trace-native native trace
bash stage4/run.sh "$data" "$engine" native-pq native evaluate

python3 stage4/summarize.py stage4/results
python3 stage3/trace_plot.py stage4/runs/{trace-reference,trace-native}/trace.sqlite \
  stage4/results/nsight-timeline.svg --labels 'TensorRT + Python' 'TensorRT + C++' --split-legacy-output
```

图表命令读取归档目录中的 `result.json` 和采集目录中的 SQLite。原生编译目标为 sm_89。检查和评估工具使用同一个 Stage 1 开发镜像。独立调用原生可执行文件可使用脚本生成的 COCO thing 清单和 TSV 图像清单，参数见 [原生入口](main.cu)。运行时依赖以 [动态链接记录](results/native-align-fixed/runtime-libraries.txt) 为准。

## 全量准确率

| TensorRT FP32 路径 | PQ | PQ_th | PQ_st | 唯一预测 ID |
| --- | ---: | ---: | ---: | ---: |
| Stage 3 Python 前后处理 | 53.186256 | 59.027261 | 44.369646 | 5,000 |
| Stage 4 C++/CUDA | 53.186256 | 59.027261 | 44.369645 | 5,000 |

5,000 张图的预测 ID 与真值完全匹配。三项 PQ 的差值均小于 0.000001 点，全部通过 0.05 点验收线。推理在 `22dc649` 生成全部 map 与 metadata。首次计分时将原始 COCO 记录的 `id` 直接传入了要求 `image_id` 的 evaluator，`ec0f13d` 修正字段映射后用同一批预测完成计分。 [推理记录](results/native-pq/console.log) · [计分记录](results/native-pq/evaluation.log) · [指标与差值](results/native-pq/result.json)

所有正式性能轮次、Nsight 采集和修正后的首轮 32 张对齐均在 `22dc649` 执行。最初的前处理和插值检查在 `7bf1b65` 执行，其实现随后沿用。RTX 4090 上的 `/home/parker/Code/MaskDINO-TensorRT-stage4-formal/stage4/runs/` 保留预测、CUDA 可执行文件和原始 SQLite。两个数据目录的全部 JPEG 与 PNG 已与 Stage 3 的 SHA-256 清单校验一致。 [数据校验](results/dataset-verification.txt)

## 计时与显存口径

三种路径各测三轮，轮换执行顺序。每轮重复第一张图预热 20 次，然后测 Stage 2 同一顺序的 500 张图，JPEG 预载到内存。CUDA events 围住网络执行。端到端用墙钟，从内存 JPEG 开始到 CPU map 与 metadata 就绪。稳态吞吐用完整测量窗口完成数除以墙钟时间。评估落盘和 PQ 计算在独立运行中执行。

三种路径均用 NVML 的 CUDA compute process 条目按宿主 PID 采样，容器使用宿主 PID 命名空间。以 20 ms 间隔记录推理进程观测峰值，涵盖网络、前后处理和 CUDA 上下文。表中以 GiB 展示三轮中位数。Stage 2 的 PyTorch allocator allocated/reserved 与 Stage 3 的整卡占用按各自历史口径保留。

### RTX 4090 性能

| 路径 | 网络 GPU P50 / P95（ms） | 端到端 P50 / P95（ms） | 吞吐（图/s） | 进程显存采样峰值（GiB） |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 37.57 / 37.81 | 48.70 / 51.86 | 20.43 | 5.19 |
| Python 前后处理 + TensorRT FP32 | 20.63 / 21.76 | 31.89 / 35.16 | 31.21 | 4.98 |
| C++/CUDA + 同一 TensorRT engine | 21.56 / 21.81 | 22.92 / 23.63 | 43.65 | 1.24 |

相对 Stage 3 的 Python/TensorRT 链路，原生链路端到端 P95 降低 **32.8%**，吞吐增加 **39.8%**，进程显存采样峰值降低 **75.1%**。网络 P95 增加 0.05 ms，P50 增加 0.93 ms。九轮均完成 500 张，失败数为 0。 [逐轮汇总](results/summary.json)

## Nsight Systems

两份 trace 分别采集同一份 32 张 COCO 清单，沿用相同的预热。NVTX 标记解码、CPU 前处理、H2D、GPU 前处理、网络、后处理和 D2H。CPU NVTX 区间显示提交与等待，GPU 活动通过 CUDA correlation ID 按提交阶段归类。

![同一 COCO 图像的 Stage 3 和 Stage 4 CPU、GPU 时间线](results/nsight-timeline.svg)

| 32 张 trace 指标 | Python + TensorRT | C++/CUDA + TensorRT |
| --- | ---: | ---: |
| GPU 网络活动均值（ms/图） | 20.14 | 20.49 |
| GPU 后处理活动均值（ms/图） | 4.34 | 0.11 |
| H2D / 后处理 / D2H 的 `cudaStreamSynchronize` 次数 | 32 / 2,591 / 32 | 0 / 0 / 32 |
| 请求区间无记录 CUDA GPU 活动的时间占比 | 24.0% | 6.9% |
| GPU 最终 D2H 活动均值（ms/图） | 0.045 | 0.044 |

原生后处理直接读取 TensorRT 输出，仅计算通过筛选的 query，融合两次插值与像素分配，GPU 上生成最终的 ID 图与 metadata。CPU 一次性提交后处理和 D2H，并在最终 D2H 等待整条 stream。原始 trace 把最终拷贝与同步包在同一个 D2H NVTX 区间内。上图使用 SQLite 的 CUDA API 事件把原生链路拆开显示。32 张中，两次 `cudaMemcpyAsync` 的 CPU 提交时间合计中位数为 **0.004 ms/图**，`cudaStreamSynchronize` 的 CPU 等待时间中位数为 **20.53 ms/图**，GPU 实际执行最终 D2H 拷贝平均为 **0.044 ms/图**。图中第一张的 17.12 ms 等待与 16.69 ms 的 GPU 网络活动重叠。 [Python 聚合](results/trace-reference/trace-summary.json) · [C++ 聚合](results/trace-native/trace-summary.json)

[Nsight 原始报告 · Python](results/trace-reference/trace.nsys-rep) · [Nsight 原始报告 · C++](results/trace-native/trace.nsys-rep)

后续原生采集会在 GUI 的 NVTX 轨直接标记 `CPU / submit D2H copies` 与 `CPU / wait for output (network + postprocess + copies)`。使用新采集绘图时省略 `--split-legacy-output`。
