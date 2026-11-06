# Stage 2 · 方形输入与 PyTorch 性能基线

本阶段固定 MaskDINO R50 panoptic 权重，建立 TensorRT 转换前的 PyTorch 对照。先统一输入形状并固定两套样本，再用全量 PQ 筛选尺寸、用离线 benchmark 量化速度与显存代价。最终采用 **1024×1024** 作为基线尺寸，**B1** 作为下一阶段优先优化的配置。[阶段决策、完整结果表与对比图](../README.md#stage-2--方形输入与-pytorch-性能基线)

## 输入与输出

为了让同一尺寸的图片直接组成 batch，JPEG 按 EXIF 方向解码为 RGB，使用 Detectron2 双线性 resize 等比缩放到指定长边。短边用 Python `round` 取整，最小为 1。右侧和底部在像素空间填 128，与官方 R50 panoptic 训练 mapper 的 `FixedSizeCrop` 默认值一致，再按官方配置归一化。mask logits 双线性插值到方形画布，裁出有效区域，恢复到原图尺寸后执行官方 panoptic 合并。推理用 FP32，关闭 autocast、TF32 和 OOM 回退。

Stage 1 官方输入采用短边 800、长边上限 1333。两阶段的 PQ 差异反映整套输入策略的变化，包括缩放和填充。

## 数据与验证用途

| 数据 | 选择方法 | 用途 |
| --- | --- | --- |
| 32 张数值验证样本 | 覆盖原图面积、宽高比、Things 数和 panoptic segment 数的极端样本，再固定种子补齐 | 先检查横、竖、方图的合成几何，再按图比较 B1 与 B2/B4/B8 的 logits、ID map 和 segment metadata |
| 500 张性能验证样本 | 按原图像素数三分位和宽高比阈值 0.9、1.1 分层，按全量比例分配配额，种子为 20261106 | 固定图像 ID 和执行顺序，用同一负载比较各尺寸及 batch |
| COCO 2017 val 全量 5,000 张 | 使用官方图像与 panoptic 标注 | 三档分辨率各跑一次 B4 PQ，核对完整预测覆盖 |

两套样本存放在 [sample.json](sample.json)，对应 `smoke_image_ids` 和 `image_ids`。500 张样本的每图平均 segment 数为 11.47，全量为 11.35。面积小于 32² 像素的非 crowd Things 占比分别为 43.47% 和 42.37%。光照未参与分层。

## 复现

### 环境和运行版本

参考环境是 RTX 4090、驱动 595.84、CUDA 13.0、PyTorch `2.10.0a0+b558c986e8.nv25.11`。沿用 [Stage 1 镜像](../stage1/README.md)与[数据及权重](../docs/experiment-environment.md)。正式运行的镜像 ID 为 `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e`。启动脚本校验数据标注与权重的 SHA-256，要求工作区干净，并将输出写到 `stage2/runs/<run-id>/`。

| 实验 | 实际运行提交 |
| --- | --- |
| 32 张数值验证、B1/B2/B4/B8 | `7b0decc6145fbac5508b8f8e6cb0cdc0173ee324` |
| 640/800/1024 全量 PQ、B1 复跑、800/1024 的 B1/B4 性能对照 | `7d8788b99327dd359b2cad29d8ae14141660d8da` |
| 1024 的 B2/B8 性能补测 | `1f96771234baefb607dceb61092fa6474fd17123` |
| 结果归档、绘图数据与统计脚本 | `stage2-pytorch-square` |

在仓库根目录执行以下命令，再在同一个 Bash 会话依次运行各实验块。填写现有数据和权重的绝对路径。`MASKDINO_RUN_ID` 加上时间戳，避免与已有结果目录冲突。

```bash
repo=$(git rev-parse --show-toplevel)
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
git worktree add --detach "${repo}-stage2-repro-${run_stamp}" stage2-pytorch-square
cd "${repo}-stage2-repro-${run_stamp}"

data=/absolute/path/to/coco2017-panoptic-val
export MASKDINO_CHECKPOINT=/absolute/path/to/maskdino_r50_panoptic.pth
export MASKDINO_IMAGE=maskdino:stage1
```

### 1. 样本和数值验证

每个尺寸先跑 B1，保存该尺寸的参考输出。同一批次对照使用相同的运行 ID。

```bash
git switch --detach 7b0decc6145fbac5508b8f8e6cb0cdc0173ee324
python3 stage2/sample.py "$data" "/tmp/stage2-sample-${run_stamp}.json"
cmp stage2/sample.json "/tmp/stage2-sample-${run_stamp}.json"

export MASKDINO_RUN_ID="probe-${run_stamp}"
for size in 640 800 1024; do
  for batch in 1 2 4 8; do
    bash stage2/run.sh "$data" probe "$size" "$batch"
  done
done
```

### 2. 三档全量 PQ

评估使用 Detectron2 的 `COCOPanopticEvaluator` 和官方验证集类别映射。评估循环包括 JPEG 读盘和 evaluator 的 PNG 编码，性能对照使用下一节的独立入口。

```bash
git switch --detach 7d8788b99327dd359b2cad29d8ae14141660d8da
export MASKDINO_RUN_ID="pq-${run_stamp}"
for size in 640 800 1024; do
  bash stage2/run.sh "$data" evaluate "$size" 4
done
```

### 3. 800 与 1024 的 B1/B4 性能对照

每组在独立容器中运行，JPEG 预读后用首批输入预热 20 个 batch，再测完整 500 张。三轮轮换配置的执行顺序。

```bash
git switch --detach 7d8788b99327dd359b2cad29d8ae14141660d8da
round=0
for order in '800-1 1024-4 800-4 1024-1' '1024-1 800-4 1024-4 800-1' '800-4 1024-1 800-1 1024-4'; do
  round=$((round + 1))
  export MASKDINO_RUN_ID="perf-${run_stamp}-r${round}"
  for pair in $order; do
    bash stage2/run.sh "$data" benchmark "${pair%-*}" "${pair#*-}"
  done
done
```

### 4. 1024 的 B2/B8 性能补测

补测提交增加批末累计墙钟记录，沿用相同的推理路径。B8 的 500 张整轮包含 62 个满批和一个 4 张尾批。前 496 张另用于 B2/B8 满批对照。

```bash
git switch --detach 1f96771234baefb607dceb61092fa6474fd17123
round=0
for order in '2 8' '8 2' '2 8'; do
  round=$((round + 1))
  export MASKDINO_RUN_ID="batch-${run_stamp}-r${round}"
  for batch in $order; do
    bash stage2/run.sh "$data" benchmark 1024 "$batch"
  done
done
```

### 5. 汇总

统计脚本位于验收 tag 中。切回该 tag 后汇总各次运行，`stage2/runs/` 中的输出可继续读取。

```bash
git switch --detach stage2-pytorch-square
python3 stage2/summarize_benchmark.py stage2/runs/perf-"${run_stamp}"-r*/benchmark/*/result.json
python3 stage2/summarize_benchmark.py stage2/runs/batch-"${run_stamp}"-r*/benchmark/*/result.json
python3 stage2/summarize_benchmark.py --prefix-images 496 \
  stage2/runs/batch-"${run_stamp}"-r*/benchmark/*/result.json
```

## 指标口径

- **网络 GPU 延迟**用 CUDA events 计量 backbone 和分割 head，按批统计 P50/P95。
- **端到端延迟**从该批的内存 JPEG 开始处理，到最后一张图的 CPU panoptic map 和 segment metadata 就绪，按批统计 P50/P95。汇总 JSON 的 `batch_e2e_*` 是批次指标，`e2e_*` 保留单请求指标。负载使用预先分好的离线批次，计时覆盖解码、前处理、H2D、推理、后处理和 D2H。
- **离线端到端吞吐**用完成图数除以整轮墙钟秒数，包含循环内的 CPU 调度开销。预读和预热在测量前完成。满批对照使用第 496 张完成后的累计墙钟。
- **峰值显存**使用预热后重置的 PyTorch CUDA allocator peak allocated 和 peak reserved，覆盖整轮。
- 结果表取各指标的三轮中位数。逐轮最小值、最大值及原始样本保存在结果目录。

## 验证结果与归档

- **数值验证。** 12 组探测均完成。32 张的 segment metadata 在各组相同。B4 相对 B1 的最坏逐图 ID map 像素差在 640 和 800 为 0，在 1024 为 0.000326%，因此全量统一采用 B4。800 和 1024 的 B4 逐 query mask logit 最大绝对差分别为 52.93 和 152.44。该比较未做 query 重配对，差异原因待定位。B1 复跑的 logits、ID map 和 metadata 逐项一致。 [逐图记录与 B1 复跑](results/probe/)
- **全量 PQ。** 三档各有 5,000 个唯一预测 ID，与真值完全匹配。640、800、1024 的 PQ 分别为 49.9662、52.0120、53.1862。 [评估日志与指标 JSON](results/evaluate/)
- **800 与 1024 的性能对照。** 12 轮均完成 500 张，无 OOM 或回退。1024/B1 的端到端 P95 为 51.93 ms，800/B1 为 37.85 ms。结合 1024 的 1.1742 PQ 优势，选择 1024 作为保留精度的部署基线。 [逐轮统计与原始数据](results/benchmark/)
- **1024 的 B2 与 B8 补测。** 六轮均完成 500 张，无 OOM 或回退。前 496 张的满批吞吐分别为 21.33 和 22.57 图/s，批次端到端 P95 分别为 97.83 和 365.06 ms。B8 尾批的网络时间为 138.90–141.55 ms，最后一张 CPU 输出于批开始后 183.28–186.11 ms 就绪。 [500 张汇总](results/batch-sweep/summary-500.json) · [496 张满批汇总](results/batch-sweep/summary-496.json) · [批次明细与日志](results/batch-sweep/)

完整运行目录保存在 4090 的 `/home/parker/Code/MaskDINO-TensorRT-stage2/stage2/runs/`。仓库归档指标、批次明细和评估日志，原始预测 JSON 和 B1 大体积张量参考保存在运行主机。
