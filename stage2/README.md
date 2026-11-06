# Stage 2 · 方形输入的 PyTorch 基线

固定 MaskDINO R50 panoptic 权重，用 COCO 2017 val 比较 640、800、1024 三档方形输入。Stage 1 的短边 800、长边上限 1333 是独立的官方基线。两阶段的 PQ 差异包含缩放策略和 padding 的变化。

## 输入与输出

JPEG 按 EXIF 方向解码为 RGB。Detectron2 的双线性 resize 保持宽高比，长边缩放到指定尺寸，短边用 Python `round` 取整，最小为 1。右侧和底部在像素空间填 128，取值与官方 R50 panoptic 训练 mapper 的 `FixedSizeCrop` 默认值一致。随后使用 checkpoint 配置的归一化参数。推理使用 FP32，关闭 autocast 和 TF32。mask logits 双线性插值到方形画布，裁出有效区域，再按官方后处理恢复到原图并合并 panoptic segment。推理路径遇到 OOM 直接报错。

## 复现

使用 [Stage 1 容器](../stage1/README.md)和[已校验的数据与权重](../docs/experiment-environment.md)。从干净的实验提交运行。可用 `MASKDINO_CHECKPOINT` 指向另一份已校验的权重文件。

```bash
data=/path/to/coco2017-panoptic-val
for size in 640 800 1024; do
  for batch in 1 2 4 8; do
    bash stage2/run.sh "$data" probe "$size" "$batch"
  done
done
```

probe 对横图、竖图和方形图的几何恢复执行合成输入检查，再用同一组 32 张 COCO 图比较每个尺寸的 B1 与 B2/B4/B8。B1 结果是后续 probe 的参考，先执行同尺寸的 B1。每次记录 PyTorch CUDA allocator 的峰值和运行环境。完整日志与大体积探测参考存放在 4090 的 `stage2/runs/<run-id>/`。如需重跑，设置新的 `MASKDINO_RUN_ID`，同一组 probe 保持同一 ID。

## 固定性能样本

[清单](sample.json)的 `image_ids` 是 500 张 JPEG 的执行顺序，`smoke_image_ids` 是 32 张探测图。性能入口按 ID 读取现有 COCO JPEG。抽样只使用数据和标注，模型结果不参与。

```bash
python3 stage2/sample.py "$data" /tmp/stage2-sample.json
cmp stage2/sample.json /tmp/stage2-sample.json
```

先按原图像素数三分位及宽高比 0.9 和 1.1 分层，按全量比例分配配额，固定种子 20261106。清单记录各层数量、均值、中位数和 P95。500 张平均 JPEG 为 164.4 kB，全量为 162.9 kB。每图平均 panoptic segment 数为 11.47 对 11.35，非 crowd Things 数为 7.39 对 7.22，标注面积小于 32² 像素的 Things 占比为 43.47% 对 42.37%。光照未参与分层。

## 32 张 GPU 探测结果

RTX 4090，驱动 595.84，CUDA 13.0，PyTorch `2.10.0a0+b558c986e8.nv25.11`。运行提交 `7b0decc6145fbac5508b8f8e6cb0cdc0173ee324`。镜像 ID 为 `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e`。

| 方形尺寸 | B1 peak allocated | B2 | B4 | B8 |
| --- | ---: | ---: | ---: | ---: |
| 640 | 1.69 GiB | 1.76 GiB | 1.88 GiB | 2.14 GiB |
| 800 | 1.98 GiB | 2.08 GiB | 2.28 GiB | 3.18 GiB |
| 1024 | 2.51 GiB | 2.67 GiB | 2.99 GiB | 5.06 GiB |

12 组全部完成，推理关闭 OOM 回退。每个尺寸的 B2、B4、B8 都与该尺寸的 B1 比较。32 张的 segment metadata 在各组均相同。B4 的最坏逐图 ID map 像素差比例在 640 和 800 为 0，在 1024 为 0.000326%。三档 B4 的 peak reserved 分别为 2.98、3.99、4.91 GiB。全量评估据此统一使用 B4。

逐 query 索引的原始 logits 存在差异。B4 在 800 和 1024 的最大 mask logit 绝对差分别为 52.93 和 152.44。比较未做 query 重配对，差异原因待定位。B4 的采用依据是部署输出对齐及显存结果。逐图差异和原始显存数值见 [probe 记录](results/probe/)。

## 全量 PQ 复现

```bash
for size in 640 800 1024; do
  bash stage2/run.sh "$data" evaluate "$size" 4
done
```

三档各跑完整 COCO 2017 val，沿用官方验证集类别映射和 Detectron2 的 `COCOPanopticEvaluator`。每组运行结束后检查预测 JSON 中的 5,000 个 ID 与真值完全匹配。评估循环包含 JPEG 读盘、推理、后处理和 evaluator 的 PNG 编码。它的耗时和 allocator 峰值用于核对运行，正式性能对照使用独立入口。

```bash
bash stage2/run.sh "$data" benchmark 800 4
```

示例命令中的尺寸和 batch 可切换。性能入口以清单顺序预读 JPEG，预热 20 个 batch，输出每个 batch 的 CUDA event 网络时间及每张图从该批开始到 CPU 输出就绪的墙钟时间。输出保留尾批的实际图像 ID 和数量。

### 全量结果

运行提交 `7d8788b99327dd359b2cad29d8ae14141660d8da`，沿用同一 RTX 4090 和镜像。首次启动时训练与验证的 Stuff 映射相等断言中断了评估，修正后使用官方验证元数据重新运行。修正只涉及 evaluator 初始化。

| 输入 | Batch | PQ | PQ_th | PQ_st | 相对 Stage 1 ΔPQ | 覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Stage 1 官方输入](../stage1/README.md) | 1 | 52.9567 | 58.9645 | 43.8884 | — | 5,000 |
| [640×640](results/evaluate/640-4/result.json) | 4 | 49.9662 | 54.8245 | 42.6330 | −2.9905 | 5,000 |
| [800×800](results/evaluate/800-4/result.json) | 4 | 52.0120 | 57.3157 | 44.0065 | −0.9447 | 5,000 |
| [1024×1024](results/evaluate/1024-4/result.json) | 4 | 53.1862 | 59.0272 | 44.3697 | +0.2295 | 5,000 |

全量 B4 三组均完成。评估循环分别耗时 121.7、172.8、248.2 秒，peak allocated 分别为 1.98、2.37、3.06 GiB。预测 JSON 和完整运行目录保留在 4090 的 `/home/parker/Code/MaskDINO-TensorRT-stage2/stage2/runs/7d8788b99327dd359b2cad29d8ae14141660d8da/evaluate/`。仓库归档每组控制台日志与原始指标 JSON。

全量评估结束后复跑各尺寸 B1。32 张图的 class logits、mask logits、ID map 和 segment metadata 与首次 B1 逐项完全一致。[复跑对照结果](results/probe/b1-repeat.json)

## 800 与 1024 性能对照

复用全量 PQ 的运行提交 `7d8788b99327dd359b2cad29d8ae14141660d8da` 和容器。四组均使用 FP32、同一份 500 张 JPEG 清单和执行顺序。每轮各配置在独立容器中运行，预读 JPEG 后，用首批输入预热 20 个 batch，再测完整 500 张。三轮轮换配置执行顺序，以减少固定顺序与机器热状态的关联。B1 和 B4 均无尾批。

```bash
round=0
for order in '800-1 1024-4 800-4 1024-1' '1024-1 800-4 1024-4 800-1' '800-4 1024-1 800-1 1024-4'; do
  round=$((round + 1))
  export MASKDINO_RUN_ID="stage2-perf-20261106-r$round"
  for pair in $order; do
    bash stage2/run.sh "$data" benchmark "${pair%-*}" "${pair#*-}"
  done
done
python3 stage2/summarize_benchmark.py stage2/runs/stage2-perf-20261106-r*/benchmark/*/result.json
```

网络 GPU 延迟是 backbone 和分割 head 的 CUDA event 时间，按 batch 统计。端到端延迟从该批内存 JPEG 开始解码，到每张图的 CPU panoptic map 和 segment metadata 就绪，按图统计，包含该批的前处理、H2D、网络、GPU 后处理和 D2H。此处使用预先分好的离线 batch。吞吐用 500 张除以测量循环墙钟时间。峰值显存统计 PyTorch CUDA allocator 的 allocated 与 reserved 范围，预热后重置峰值。

| 输入 | Batch | 网络 GPU P50/P95（ms/batch） | 端到端 P50/P95（ms/图） | 吞吐（图/s） | 峰值 allocated / reserved（GiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 800×800 | 1 | 26.49 / 26.77 | 35.02 / 37.85 | 28.35 | 2.07 / 3.92 |
| 800×800 | 4 | 87.61 / 88.14 | 115.40 / 124.74 | 32.72 | 2.37 / 4.83 |
| 1024×1024 | 1 | 37.77 / 37.88 | 48.79 / 51.93 | 20.38 | 2.58 / 4.71 |
| 1024×1024 | 4 | 138.83 / 139.37 | 174.92 / 186.41 | 21.82 | 3.06 / 4.91 |

表中每项为三轮中位数，范围与逐轮统计见 [汇总 JSON](results/benchmark/summary.json)。12 轮各完成 500 张，顺序均与清单一致，未发生 OOM 或回退。[原始延迟和日志](results/benchmark/)

1024 相对 800 获得 1.1742 PQ，B1 端到端 P95 增加 14.08 ms，吞吐由 28.35 降到 20.38 图/s。1024 从 B1 到 B4 的离线吞吐提高约 7%，单请求 P95 增至约 3.6 倍。若没有更紧的延迟预算，后续可用 1024/B1 作为优先优化的配置，1024/B4 作为离线吞吐对照。
