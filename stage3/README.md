# Stage 3 · TensorRT 单请求网络后端

固定 MaskDINO R50 panoptic、1024×1024、batch=1、单 slot。前处理、归一化、panoptic 后处理和 COCO evaluator 共用 Stage 2 实现。PyTorch FP32 与 TensorRT 均关闭 TF32。

## 验收标准

- 复用 Stage 2 的 32 张验证清单。逐图报告 logits、masks、boxes 的最大绝对误差、平均绝对误差和相对 L2。张量形状须一致且所有值有限。当 mask 的直接相对 L2 大于 0.02 时，使用下采样 mask 余弦距离与类别概率距离做匈牙利匹配，附上配对后误差。匹配只用于诊断，后处理仍使用原始输出。
- 类别图比较包含 void，并区分 thing 和 stuff。对同类 segment 按 IoU 从高到低一对一配对，记录原始 metadata 是否相同、每个匹配的 IoU 和未匹配数量。FP32 逐图类别一致率至少 99.9%，所有 segment 均配对且 IoU 至少 0.98。异常保存原始输入、输出张量、panoptic map、metadata 和差异图。
- 全量 5,000 张预测 ID 唯一且与真值集合一致。FP32 相对 PyTorch B1 的 PQ、PQ_th、PQ_st 各项下降最多 0.05 点。Stage 2 B4 的 PQ 53.1862 单独列作参考。
- 每种后端用相同顺序跑 Stage 2 的 500 张性能清单，每轮预热 20 次，共三轮。内存 JPEG 到 CPU panoptic map 与 metadata 就绪为端到端边界。GPU 网络时间用 CUDA events。整卡 NVML 20 ms 采样涵盖 TensorRT 内存及桌面占用，记录观测峰值。
- Nsight Systems 单独采集，NVTX 覆盖解码、前处理、H2D、GPU 归一化、网络、后处理、D2H。

## 复现入口

在 4090 上从干净的提交运行。准备 Stage 2 镜像及数据，设置 `MASKDINO_CHECKPOINT` 为权重路径。每次运行用新的 ID。日志和产物位于 `stage3/runs/<ID>/`。容器内可通过 `/project/stage3/runs/<ID>/` 读取之前的产物。

```bash
export MASKDINO_IMAGE=maskdino:stage1
export MASKDINO_CHECKPOINT=/path/to/maskdino_r50_panoptic.pth
data=/path/to/coco2017-panoptic-val
runs=/project/stage3/runs

# 复现时各运行 ID 必须未使用。
bash stage3/run.sh "$data" constants-raw repro-constants
bash stage3/run.sh "$data" constants-fixed repro-constants --cpu-constants
bash stage3/run.sh "$data" export-fp32-raw export --backend fp32
bash stage3/run.sh "$data" export-fp32-fixed export --backend fp32 --cpu-constants
bash stage3/run.sh "$data" msda plugin-test
bash stage3/run.sh "$data" build-fp32 build --backend fp32 --onnx "$runs/export-fp32-fixed/network.onnx"
bash stage3/run.sh "$data" align-fp32 align --backend fp32 --engine "$runs/build-fp32/network.engine"
bash stage3/run.sh "$data" pt-pq evaluate
bash stage3/run.sh "$data" fp32-pq evaluate --backend fp32 --engine "$runs/build-fp32/network.engine"
python3 stage3/summarize.py stage3/runs/{pt-pq,fp32-pq}/result.json

for round in 1 2 3; do
  if (( round == 2 )); then order=(fp32 pytorch); else order=(pytorch fp32); fi
  for backend in "${order[@]}"; do
    args=()
    if [[ "$backend" == fp32 ]]; then args=(--engine "$runs/build-fp32/network.engine"); fi
    bash stage3/run.sh "$data" "perf-$backend-r$round" benchmark --backend "$backend" "${args[@]}"
  done
done

```

无补丁的常量复现和完整导出预期以 SIGSEGV 退出，分别保留日志。每次运行均记录提交、镜像 ID、命令和资产哈希。完整 COCO JPEG 与 PNG 的 SHA-256 清单可在数据根目录用以下命令生成，然后与 [环境记录](results/environment.json) 的清单摘要对照。

```bash
for asset in val2017 panoptic_val2017; do
  find "$asset" -maxdepth 1 -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "/tmp/stage3-$asset.sha256"
  wc -l "/tmp/stage3-$asset.sha256"
  sha256sum "/tmp/stage3-$asset.sha256"
done
```

## RTX 4090 运行记录

- PyTorch B1 全量 PQ 在 `9d79baa` 运行。完整覆盖 5,000 个预测 ID，PQ / PQ_th / PQ_st 为 53.186252 / 59.027259 / 44.369638。该准确率对照使用 PyTorch 默认 stream。正式性能对照的两个后端均使用同样的单条显式 stream。 [PyTorch B1 原始记录](results/pt-pq/)
- FP32 导出、CUDA 常量复现与插件独立验证在 `b4c5eb3` 运行。未修复的最小复现于 ONNX Slice shape inference 崩溃，完整导出同样触发 SIGSEGV。修复仅把导出器访问的常量属性移到 CPU，模型和输入继续位于 GPU。最小复现移动 2 个属性，完整 FP32 导出移动 21 个。 [最小复现日志](results/constants-raw/console.log) · [完整导出失败日志](results/export-fp32-raw/console.log) · [修复后导出信息](results/export-fp32-fixed/export.json)
- NVIDIA MSDA v2 对随机样本的 FP32 输出与上游 CUDA 算子逐元素一致。样本使用 4 个尺度、21,760 个空间位置、8 个 head、每 head 32 通道以及 300 个 query，采样点包含越界坐标。CUDA 对 grid-sample 最大绝对误差为 \(6.67\times 10^{-7}\)。 [NVIDIA 官方插件](https://github.com/NVIDIA/TensorRT/tree/main/plugin/multiscaleDeformableAttnPlugin) · [数值验证](results/msda/result.json)
- FP32 engine 构建和 32 张对齐在 `dd6213a` 运行。固定输入与输出全部为 FP32，engine inspector 确认 6 个 encoder 和 9 个 decoder PluginV3 层，输入 shapes 和 starts 为 INT32。禁用 TF32 与辅助 stream，workspace 上限 4 GiB。 [构建信息](results/build-fp32/build.json) · [逐层信息](results/build-fp32/layers.json)
- 32 张全部通过，metadata 全部一致。最差类别一致率 99.99873%，同类 segment 最小 IoU 0.999608。3 张图有一对 query 互换，直接 mask 相对 L2 为 4.33%–5.55%，匹配后为约 0.0003%–0.0013%。 [逐图张量及后处理数据](results/align-fp32/result.json)
- 首次 TensorRT PQ 启动在第一张图失败。释放 PyTorch 网络显存时删除了后处理读取的 `sem_seg_head.num_classes`。将网络模块移至 CPU 后保留同一个后处理方法与元数据，重跑通过。TensorRT FP32 全量覆盖 5,000 个唯一 ID，PQ / PQ_th / PQ_st 为 53.186256 / 59.027261 / 44.369646。 [首次失败日志](results/fp32-pq-initial-failure/console.log) · [重跑结果](results/fp32-pq-rerun/result.json)

### 混合精度调查

失败样本、缩小 FP16 范围的对照、复现命令与未验证假设单独归档。 [混合精度问题记录](mixed-precision.md)

### 全量准确率

| 1024×1024 后端 | Batch | PQ | PQ_th | PQ_st | 唯一预测 ID |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 1 | 53.186252 | 59.027259 | 44.369638 | 5,000 |
| TensorRT FP32 | 1 | 53.186256 | 59.027261 | 44.369646 | 5,000 |
| Stage 2 PyTorch FP32，历史参考 | 4 | 53.186242 | 59.027230 | 44.369656 | 5,000 |

TensorRT FP32 的三项 PQ 指标均通过验收，与 PyTorch B1 的差值各小于 0.00001 点。 [FP32 验收记录](results/pq-fp32-gate.json) · [Stage 2 B4 全量结果](../stage2/results/evaluate/1024-4/result.json)

完整实验位于 4090 的 `/home/parker/Code/MaskDINO-TensorRT-stage3/stage3/runs/`。其中保留 ONNX、engine、预测 JSON、异常图张量和两套数据文件哈希清单。仓库归档日志、配置、engine inspector、数值指标、差异图与原始计时样本。

### 性能

每种后端三轮均完成 500 张，失败数为 0。网络 GPU P50/P95 用 CUDA events 计时。端到端从内存 JPEG 到 CPU panoptic map 和 metadata。显存使用整卡 NVML 20 ms 采样，表中为三轮观测峰值的中位数，包含桌面进程和 TensorRT 分配。

| 后端 | 网络 P50 / P95（ms） | 端到端 P50 / P95（ms） | 吞吐（图/s） | 整卡采样峰值（GiB） |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 37.58 / 37.75 | 48.68 / 51.87 | 20.44 | 5.79 |
| TensorRT FP32 | 20.65 / 21.79 | 31.91 / 35.18 | 31.18 | 5.59 |

- **TensorRT 转换收益。** FP32 转换使网络 P95 减少 15.97 ms（42.3%），端到端 P95 减少 16.69 ms（32.2%），吞吐增加 10.74 图/s（52.5%）。

两组 FP32 性能运行均使用 `18986d2`。 [验收汇总与逐轮范围](results/summary.json)

### Nsight Systems 基线

两组 FP32 trace 均从 `d56ac7a` 运行，各预热 20 次，使用同一份 32 张清单，并与正式性能计时分开采集。TensorRT trace 使用与对齐、PQ、性能测试相同的 engine。 [PyTorch 采集](results/trace-pytorch-rerun/) · [TensorRT 采集](results/trace-fp32/)

![Nsight Systems FP32 时间线](results/nsight-fp32-timeline.svg)

图由 Nsight Systems 导出的 SQLite 原始事件绘制，展示两组预热后的同一张 COCO 图 149770，横轴共用毫秒尺度。Host NVTX 表示 CPU 所在阶段，GPU activity 表示该进程的 CUDA kernel、copy 和 memset 区间并集。下表统计全部 32 张。

| 独立 trace 指标 | PyTorch FP32 | TensorRT FP32 |
| --- | ---: | ---: |
| 请求区间无记录 CUDA GPU 活动的时间占比 | 26.4% | 24.1% |
| 按 CUDA API 关联到网络提交的 GPU 活动，平均 ms/图 | 33.74 | 19.79 |
| 按 CUDA API 关联到后处理提交的 GPU 活动，平均 ms/图 | 4.33 | 4.34 |
| 网络区间的 `cudaStreamSynchronize` 次数 | 2,432 | 0 |
| 后处理区间的 `cudaStreamSynchronize` 次数 | 2,591 | 2,591 |
| 最终 D2H 区间内 GPU 活动，平均 ms/图 | 0.045 | 0.045 |

TensorRT 异步提交网络，GPU 网络执行延续到后处理的 NVTX 区间，后处理主机区间中位数 24.00 ms 包含这段等待。两组后处理自身的 GPU 活动均约 4.3 ms/图。CPU 解码和前处理平均合计约 5.4 ms/图，其间无该进程的 CUDA GPU 活动。后处理每图约 81 次 stream 同步，源代码中的 GPU 布尔筛选和逐 segment 的 `.item()` 是可检查的同步点。 [PyTorch 聚合](results/trace-pytorch-rerun/trace-summary.json) · [TensorRT 聚合](results/trace-fp32/trace-summary.json)

原始 `.nsys-rep` 和 SQLite 位于 4090 对应运行目录，仓库内保存报告哈希、`nsys stats` 和逐阶段聚合。

```bash
bash stage3/run.sh "$data" trace-pytorch-rerun trace --backend pytorch
bash stage3/run.sh "$data" trace-fp32 trace --backend fp32 --engine "$runs/build-fp32/network.engine"
for id in trace-pytorch-rerun trace-fp32; do
  capture="stage3/runs/$id"
  nsys export --type=sqlite --output="$capture/trace.sqlite" "$capture/trace.nsys-rep"
  nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "$capture/trace.sqlite"
  python3 stage3/trace_summary.py "$capture/trace.sqlite"
done
python3 stage3/trace_plot.py stage3/runs/{trace-pytorch-rerun,trace-fp32}/trace.sqlite \
  stage3/results/nsight-fp32-timeline.svg
python3 stage3/summarize.py stage3/runs/{pt-pq,fp32-pq}/result.json \
  stage3/runs/perf-{pytorch,fp32}-r{1,2,3}/result.json > stage3/results/summary.json
python3 stage3/plot.py stage3/results
```
