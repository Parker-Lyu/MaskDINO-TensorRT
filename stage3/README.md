# Stage 3 · 将网络转换为 TensorRT

固定 MaskDINO R50 panoptic、1024×1024、batch=1、单 slot。前处理、归一化、panoptic 后处理和 COCO evaluator 共用 Stage 2 实现。PyTorch FP32 参考和首轮 TensorRT FP32 验收关闭 TF32，另对照两种允许 TF32 的策略。

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

下表使用 Stage 3 同轮复测的 PyTorch 和 TensorRT，量化网络后端的替换收益。每种后端三轮均完成 500 张，失败数为 0。网络 GPU P50/P95 用 CUDA events 计时。端到端从内存 JPEG 到 CPU panoptic map 和 metadata。显存使用整卡 NVML 20 ms 采样，表中为三轮观测峰值的中位数，包含桌面进程和 TensorRT 分配。

| 后端 | 网络 P50 / P95（ms） | 端到端 P50 / P95（ms） | 吞吐（图/s） | 整卡采样峰值（GiB） |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 37.58 / 37.75 | 48.68 / 51.87 | 20.44 | 5.79 |
| TensorRT FP32 | 20.65 / 21.79 | 31.91 / 35.18 | 31.18 | 5.59 |

- **TensorRT 转换收益。** FP32 转换使网络 P95 减少 15.97 ms（42.3%），端到端 P95 减少 16.69 ms（32.2%），吞吐增加 10.74 图/s（52.5%）。

两组 FP32 性能运行均使用 `18986d2`。 [验收汇总与逐轮范围](results/summary.json)

![Stage 1–3 全量 COCO PQ 与 Stage 3 同轮后端性能对照](results/comparison.svg)

### Nsight Systems 基线

两组 FP32 trace 均从 `d56ac7a` 运行，各预热 20 次，使用同一份 32 张清单，并与正式性能计时分开采集。TensorRT trace 使用与对齐、PQ、性能测试相同的 engine。 [PyTorch 采集](results/trace-pytorch-rerun/) · [TensorRT 采集](results/trace-fp32/)

![Nsight Systems FP32 时间线](results/nsight-fp32-timeline.svg)

图由 Nsight Systems 导出的 SQLite 原始事件绘制，展示两组预热后的同一张 COCO 图 149770，横轴共用毫秒尺度。第一行按 NVTX 阶段着色，红色覆盖记录到的 CUDA 同步 API 区间。第二行单独展示同一批同步调用。GPU 活动用 CUDA correlation ID 关联到提交它的 NVTX 阶段，灰色表示未归属的事件。三条轨道可在同一时刻重叠，图中数值不能相加得到端到端时间。

TensorRT 在该图的 CPU 后处理 NVTX 区间为 20.60 ms，其中 CUDA 同步 API 占 19.82 ms。整个区间与此前由网络提交的 GPU 活动重叠 17.25 ms，由后处理提交的 GPU 活动为 2.38 ms。PyTorch 的对应数值为 2.99 ms、2.29 ms、0.00 ms 和 2.36 ms。第一行的绿色只表示后处理 NVTX 区间中未被已记录的 CUDA 同步调用覆盖的墙钟时间。下表统计全部 32 张。

| 独立 trace 指标 | PyTorch FP32 | TensorRT FP32 |
| --- | ---: | ---: |
| 请求区间无记录 CUDA GPU 活动的时间占比 | 26.4% | 24.1% |
| 按 CUDA API 关联到网络提交的 GPU 活动，平均 ms/图 | 33.74 | 19.79 |
| 按 CUDA API 关联到后处理提交的 GPU 活动，平均 ms/图 | 4.33 | 4.34 |
| 网络区间的 `cudaStreamSynchronize` 次数 | 2,432 | 0 |
| 后处理区间的 `cudaStreamSynchronize` 次数 | 2,591 | 2,591 |
| 最终 D2H 区间内 GPU 活动，平均 ms/图 | 0.045 | 0.045 |

TensorRT 异步提交网络，GPU 网络执行延续到后处理的 NVTX 区间，后处理主机区间中位数 24.00 ms 包含这段等待。两组后处理自身的 GPU 活动均约 4.3 ms/图。CPU 解码和前处理平均合计约 5.4 ms/图，其间无该进程的 CUDA GPU 活动。后处理每图约 81 次 stream 同步，源代码中的 GPU 布尔筛选和逐 segment 的 `.item()` 是可检查的同步点。 [PyTorch 聚合](results/trace-pytorch-rerun/trace-summary.json) · [TensorRT 聚合](results/trace-fp32/trace-summary.json)

仓库保存两份可直接在 Nsight Systems 2025.5.1 或更新版本中打开的原始报告。每份包含同一清单的全部 32 张图，采集目录同时保存报告哈希、`nsys stats` 和逐阶段聚合。SQLite 导出位于 4090 对应运行目录。 [PyTorch FP32 原始报告](results/trace-pytorch-rerun/trace.nsys-rep) · [TensorRT FP32 原始报告](results/trace-fp32/trace.nsys-rep)

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
  stage3/results/nsight-fp32-timeline.svg --show-cpu-sync
python3 stage3/summarize.py stage3/runs/{pt-pq,fp32-pq}/result.json \
  stage3/runs/perf-{pytorch,fp32}-r{1,2,3}/result.json > stage3/results/summary.json
python3 stage3/plot.py stage3/results
```

#### 带明确 CPU 阶段名的重采报告

从 `de2e61f` 重采两组 FP32，沿用同一份 32 张清单、20 次预热和同一 TensorRT engine。Nsight Systems 的 NVTX Range Summary 现在将两段标成 `CPU / network call (GPU may continue after return)` 和 `CPU / postprocess (includes CUDA waits)`。前者覆盖主机调用，后者覆盖主机后处理代码及其同步等待。查看单张图时，可把 NVTX、CUDA API 和 CUDA stream 轨道同时展开。上方时间线和统计表对应原始验收采集。

- [PyTorch FP32 新版原始报告](results/trace-pytorch-nvtx/trace.nsys-rep) · [统计目录](results/trace-pytorch-nvtx/)
- [TensorRT FP32 新版原始报告](results/trace-fp32-nvtx/trace.nsys-rep) · [统计目录](results/trace-fp32-nvtx/)

```bash
bash stage3/run.sh "$data" trace-pytorch-nvtx trace --backend pytorch
bash stage3/run.sh "$data" trace-fp32-nvtx trace --backend fp32 --engine "$runs/build-fp32/network.engine"
```

### GPU 分阶段原始报告

从 `d88ff9c` 在 RTX 4090 重采 Stage 3 的 Python 前后处理 + TensorRT FP32 链路。固定原 engine、1024×1024、B1、单 slot、同一份 32 张清单和 20 次预热。NVTX 标签为 `TensorRT inference`、`postprocess` 和 `D2H`，与 Stage 4 最新原始报告使用同样的阶段名。Nsight 的 GPU 投影跨度包含同阶段 GPU 操作之间的空隙。

| 阶段 | GPU 投影跨度中位数（ms/图） |
| --- | ---: |
| TensorRT inference | 19.879 |
| postprocess | 4.737 |
| D2H | 0.051 |

[新版 Nsight 原始报告](results/trace-fp32-gpu-projection/trace.nsys-rep) · [Nsight GPU Projection 统计](results/trace-fp32-gpu-projection/nsys-gpu-projection.txt)

使用 Stage 4 的对照入口可复现同一条 Stage 3 链路与统计。设置 `data`、`engine` 和 `MASKDINO_CHECKPOINT` 后运行：

```bash
bash stage4/run.sh "$data" "$engine" trace-stage3-gpu-projection reference trace
```

## B1/S1 精度策略筛选

使用相同的 PyTorch 前后处理和 PyTorch FP32 B1 参考，对比历史的 FP32/TF32 off、FP32/TF32 on 和模型级 autocast FP16/TF32 on。两个 FP32 engine 使用相同的 FP32 ONNX。混合 engine 复用 `mixed-scope=all` 的导出，图内保留显式 FP32 区域，三个 engine 的输入和三个输出均为 FP32。两种 on 构建允许 FP32 算子选择 TF32 tactic，使用 4 GiB workspace、零辅助 stream 和相同的 TensorRT 10.14.1.48。Inspector 中 FP32/on 有 245 个 tactic 名称含 `tf32`，混合/on 有 125 个名称含 `tf32`、112 个名称含 `f16f16`。 [FP32/on 构建](results/precision/build-fp32-on/) · [混合/on 构建](results/precision/build-autocast-on/) · [混合 ONNX 导出](results/export-mixed-rerun/export.json)

逐图统一按类别一致率 ≥99.9%、segment 全部配对、同类 IoU ≥0.98 统计。全量三项均以 [PyTorch FP32 B1](results/pt-pq/result.json) 为参照，各项最多下降 0.05 点。

| Engine | 逐图通过 | 最差类别一致率 | PQ / PQ_th / PQ_st | ΔPQ / ΔPQ_th / ΔPQ_st（点） |
| --- | ---: | ---: | --- | --- |
| FP32 / TF32 off | 32/32 | 99.9987% | 53.186256 / 59.027261 / 44.369646 | +0.000005 / +0.000002 / +0.000008 |
| FP32 / TF32 on | 27/32 | 99.5868% | 53.174509 / 59.001049 / 44.379732 | −0.011743 / −0.026210 / +0.010094 |
| FP16 混合精度 / TF32 on | 16/32 | 91.6320% | 53.168045 / 59.007167 / 44.354277 | −0.018206 / −0.020092 / −0.015360 |

三组均覆盖 **5,000** 张且全量 PQ 达标。FP32/on 的 5 张逐图失败只涉及类别一致率。模型级 autocast 按原 mixed backend 的阈值（99% 和 0.90）通过 26/32，按三组统一阈值通过 16/32。两种 on 策略均保留为性能候选，沿用逐图硬门槛时两种都未通过。 [FP32/on 逐图](results/precision/precision-align-on/result.json) · [FP32/on 全量](results/precision/precision-pq-on/result.json) · [混合/on 逐图](results/precision/precision-align-autocast-on/result.json) · [混合/on 全量](results/precision/precision-pq-autocast-on/result.json)

### 构建与复现

复用 RTX 4090 上已有的两个 on engine。构建日志记录 FP32/on 运行提交 `7fb2a97`，混合/on 运行提交 `81a7d61`。Python 逐图及 COCO 评估在 `acaf623` 上完成，使用相同镜像 `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e`。原始 ONNX 和 engine 见 [Stage 4 运行来源](../stage4/README.md#运行来源与复现)，诊断张量留在 4090 的 `/home/parker/Code/MaskDINO-TensorRT-precision/repo/stage3/runs/`，构建校验值见各组 `build.json`，容器内 `NVIDIA_TF32_OVERRIDE` 未设置。

沿用上方 `data`、`runs` 和权重路径。分别将现有的 FP32 与 autocast ONNX 放入 `stage3/runs/precision-input-{fp32,mixed}/network.onnx`，可重建两个候选。若已有相同 engine，复制 `network.engine` 和匹配的 `build.json` 到相应构建目录，直接从 `align` 开始。

```bash
for backend in fp32 mixed; do
  build="precision-build-$backend-on"
  bash stage3/run.sh "$data" "$build" build --backend "$backend" --batch 1 \
    --tf32 on --onnx "$runs/precision-input-$backend/network.onnx"
  engine="$runs/$build/network.engine"
  bash stage3/run.sh "$data" "precision-align-$backend-on" align --backend "$backend" --engine "$engine"
  bash stage3/run.sh "$data" "precision-pq-$backend-on" evaluate --backend "$backend" --engine "$engine"
done
```

逐图命令先保存完整结果，再按对应 backend 的阈值返回失败。本轮以逐图差异作为诊断，继续独立跑全量 PQ，并在表中用统一的 99.9%/0.98 阈值重计通过数。原生 B1/S1 性能配对见 [Stage 4](../stage4/README.md#b1s1-精度策略对照)。
