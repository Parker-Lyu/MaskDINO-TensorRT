# Stage 3 混合精度问题记录

正式验收和后续优化基线采用 PyTorch FP32 与 TensorRT FP32。较大范围的 FP16 候选在逐图对齐中出现明显误差，后续需进一步定位传播路径。以下保留失败样本、缩小范围后的对照和复现入口。

逐图门槛为类别一致率至少 99%，所有 segment 可配对且同类 IoU 至少 0.90。进入全量后，PQ、PQ_th、PQ_st 相对 PyTorch B1 各最多下降 0.20 点。使用 [Stage 3 的环境与数据](README.md)，`data` 和 `runs` 与该页复现命令相同。

## 逐图筛选

| FP16 范围 | 通过图数 | 最低类别一致率 | 结果 |
| --- | ---: | ---: | --- |
| 模型级 autocast，沿用上游显式 FP32 区域 | 25 / 32 | 93.5712% | 停在逐图验收 |
| 仅 backbone autocast | 27 / 32 | 93.5900% | 停在逐图验收 |
| 仅最终 mask 投影 | 32 / 32 | 99.9867% | 进入全量验收 |

模型级 autocast 的第一次导出在 MSDA symbolic 处失败。Cast 节点缺少静态 shape 元数据，修复按固定 head 数和通道数声明 MSDA 几何，并校验导出器能推断出的维度。修复后，该候选的 15 个 MSDA 插件仍是 FP32。只将 backbone 降为 FP16 后，模型级 autocast 的 7 张失败图中有 4 张再次失败。对 435081，两种候选的类别一致率均约 93.6%。这说明 backbone 精度变化足以复现该图的大部分误差。逐层误差传播路径尚未做张量归因。 [首次导出失败](results/export-mixed-initial-failure/console.log) · [模型级 autocast 差异](results/align-mixed/) · [仅 backbone 差异](results/align-mixed-backbone/)

最终候选只在 ONNX 末端 mask Einsum 的两个输入前插入 FP16 Cast，复用原有的 FP32 输出 Cast。engine inspector 显示该层选用了 `sm80_xmma_gemm_f16f16_f16f16_f16` tactic，输出边界仍是 FP32。32 张均通过，最小 segment IoU 为 0.998230。 [导出与检查](results/export-mixed-mask-projection-rerun/) · [engine 逐层信息](results/build-mixed-mask-projection/layers.json) · [对齐结果](results/align-mixed-mask-projection/result.json)

```bash
bash stage3/run.sh "$data" export-mixed-rerun export --backend mixed --cpu-constants
bash stage3/run.sh "$data" build-mixed build --backend mixed --onnx "$runs/export-mixed-rerun/network.onnx"
bash stage3/run.sh "$data" align-mixed align --backend mixed --engine "$runs/build-mixed/network.engine"

bash stage3/run.sh "$data" export-mixed-backbone export --backend mixed --cpu-constants --mixed-scope backbone
bash stage3/run.sh "$data" build-mixed-backbone build --backend mixed --onnx "$runs/export-mixed-backbone/network.onnx"
bash stage3/run.sh "$data" align-mixed-backbone align --backend mixed --engine "$runs/build-mixed-backbone/network.engine"

bash stage3/run.sh "$data" export-mixed-mask-projection-rerun export --backend mixed --cpu-constants --mixed-scope mask-projection
bash stage3/run.sh "$data" build-mixed-mask-projection build --backend mixed --onnx "$runs/export-mixed-mask-projection-rerun/network.onnx"
mixed_engine="$runs/build-mixed-mask-projection/network.engine"
bash stage3/run.sh "$data" align-mixed-mask-projection align --backend mixed --engine "$mixed_engine"
bash stage3/run.sh "$data" mixed-pq evaluate --backend mixed --engine "$mixed_engine"
python3 stage3/summarize.py stage3/runs/{pt-pq,mixed-pq}/result.json
for round in 1 2 3; do
  bash stage3/run.sh "$data" "perf-mixed-r$round" benchmark --backend mixed --engine "$mixed_engine"
done
```

## 后续排查

- 模型级 autocast 和仅 backbone FP16 的共同失败图为 449661、49091、435081、36494。最差图 435081 在两组的类别一致率都约为 93.6%，与仅发生 query 换位的 FP32 情况不同。模型级 autocast 的该图 mask 相对 L2 在匹配后仍约 8.9%。原始张量保存在 4090 的对应 `align-mixed*` 运行目录，类别与 ID 差异图已归档。
- 建议先用同一输入在 PyTorch 比较 FP32 和仅 backbone FP16 的 proposal 排名，记录 TopK 索引交集及第 300/301 名的分数差，再固定 FP32 的索引做反事实验证。TopK 截断是否放大误差仍待实测。
- 最终 mask 投影 FP16 作为局部试验通过了逐图与全量 PQ。其 5,000 张的 PQ / PQ_th / PQ_st 为 53.188202 / 59.030471 / 44.369681。 [全量结果](results/mixed-pq/result.json)
- **混合精度额外收益。** 相对 TensorRT FP32，网络 P95 减少 0.223 ms，端到端 P95 减少 0.024 ms，吞吐减少 0.086 图/s。两者的端到端 P95 逐轮范围分别为 35.14–35.20 ms（混合版）与 35.04–35.19 ms（FP32），本实验未观察到有实际意义的额外端到端收益。 [汇总与逐轮范围](results/summary-mixed-investigation.json)
- 混合版 Nsight trace 中，两处输入 Cast 合计约 0.081 ms/图，FP16 GEMM 约 0.068 ms/图，输出转回 FP32 约 0.104 ms/图。Cast 约占这段 GPU 时间的 73%。 [混合版 Nsight 数据](results/trace-mixed/nsys-stats.txt)
- 补采的 TensorRT FP32 trace 中，最终 mask GEMM 约 0.233 ms/图，随后的 FP32 输出 kernel 约 0.142 ms/图。两者合计约 0.376 ms/图，混合版同一路径约 0.253 ms/图，局部节省约 0.123 ms/图。该差值来自独立 trace 的 kernel 平均值。 [FP32 Nsight 数据](results/trace-fp32/nsys-stats.txt)
