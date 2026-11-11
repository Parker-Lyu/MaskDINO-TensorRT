# Stage 3 Mixed-Precision Investigation

**English** · [简体中文](mixed-precision.zh-CN.md)

Formal acceptance and the baseline for further optimization use PyTorch FP32 and TensorRT FP32. Candidates with broader FP16 coverage produced substantial per-image differences, and the propagation path needs further investigation. The failing samples, narrower-scope comparisons, and reproduction entry points are below.

The per-image gate requires at least 99% category agreement, a match for every segment, and same-class IoU of at least 0.90. For full evaluation, PQ, PQ_th, and PQ_st may each drop by at most 0.20 points against PyTorch B1. Use the [Stage 3 environment and data](README.md) and the same `data` and `runs` as in its reproduction commands.

## Per-Image Screening

| FP16 scope | Images passed | Lowest category agreement | Outcome |
| --- | ---: | ---: | --- |
| Model-level autocast with upstream explicit FP32 regions | 25 / 32 | 93.5712% | Stopped at the per-image gate |
| Backbone-only autocast | 27 / 32 | 93.5900% | Stopped at the per-image gate |
| Final mask projection only | 32 / 32 | 99.9867% | Advanced to full evaluation |

The first model-level autocast export failed in the MSDA symbolic. Cast nodes lacked static shape metadata. The fix declares MSDA geometry using fixed head and channel counts and checks the dimensions that the exporter can infer. After the fix, all 15 MSDA plugins in the candidate still use FP32. With FP16 limited to the backbone, 4 of the 7 images that failed model-level autocast failed again. For 435081, both candidates reach about 93.6% category agreement. The backbone precision change alone reproduces most of the error on that image. The layer-by-layer propagation path has yet to be attributed using tensors. [First export failure](results/export-mixed-initial-failure/console.log) · [Model-level autocast differences](results/align-mixed/) · [Backbone-only differences](results/align-mixed-backbone/)

The final candidate inserts FP16 Casts before the two inputs of the final ONNX mask Einsum and reuses the existing FP32 output Cast. Engine Inspector shows the `sm80_xmma_gemm_f16f16_f16f16_f16` tactic for this layer with an FP32 output boundary. All 32 images pass, with a minimum segment IoU of 0.998230. [Export and checks](results/export-mixed-mask-projection-rerun/) · [Engine layer details](results/build-mixed-mask-projection/layers.json) · [Alignment results](results/align-mixed-mask-projection/result.json)

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

## Follow-Up Investigation

- Images 449661, 49091, 435081, and 36494 fail both model-level autocast and backbone-only FP16. The worst image, 435081, reaches about 93.6% category agreement in both configurations. The FP32 case involved only a query swap. With model-level autocast, matched mask relative L2 on this image is still about 8.9%. Raw tensors are in the corresponding `align-mixed*` run directories on the 4090. Category and ID difference images are archived.
- First compare proposal rankings for FP32 and backbone-only FP16 in PyTorch on the same input. Record the overlap of TopK indices and the score gap between ranks 300 and 301, then fix the FP32 indices for a counterfactual test. Whether the TopK cutoff amplifies error remains to be measured.
- The targeted final mask projection FP16 experiment passed both per-image checks and full PQ. Its 5,000-image PQ / PQ_th / PQ_st was 53.188202 / 59.030471 / 44.369681. [Full results](results/mixed-pq/result.json)
- **Incremental mixed-precision gains.** Against TensorRT FP32, network P95 falls by 0.223 ms, end-to-end P95 falls by 0.024 ms, and throughput falls by 0.086 images/s. End-to-end P95 across runs ranges from 35.14–35.20 ms for mixed precision and 35.04–35.19 ms for FP32. The experiment observed no meaningful incremental end-to-end gain. [Summary and per-run ranges](results/summary-mixed-investigation.json)
- In the mixed-precision Nsight trace, the two input Casts total about 0.081 ms/image, FP16 GEMM takes about 0.068 ms/image, and the cast back to FP32 takes about 0.104 ms/image. Casts take about 73% of GPU time in this region. [Mixed-precision Nsight data](results/trace-mixed/nsys-stats.txt)
- In the follow-up TensorRT FP32 trace, the final mask GEMM takes about 0.233 ms/image and the following FP32 output kernel takes about 0.142 ms/image. They total about 0.376 ms/image. The same path in the mixed version takes about 0.253 ms/image, saving about 0.123 ms/image locally. This difference uses kernel means from separate traces. [FP32 Nsight data](results/trace-fp32/nsys-stats.txt)
