# Stage 6 · TensorRT Layer Profiling, Targeted FP16, and Diagnosing Output Differences

**English** · [简体中文](README.zh-CN.md)

Stages 1–5 reproduced MaskDINO R50 panoptic accuracy, converted the network to fixed-shape TensorRT, implemented C++/CUDA pre- and postprocessing, and added concurrency with multiple CUDA streams. This stage keeps the pipeline architecture and addresses three questions.

1. **Where does the network spend its time?** Network structure and FLOPs alone cannot identify the bottlenecks.
2. **Can lowering precision only for hotspot operators improve performance with less numerical risk?** Include casts, reformats, and TensorRT tactic selection in the actual cost.
3. **How can we attribute A/B differences in performance and outputs to the targeted change?** A single ONNX node edit can change multiple operators in the built engine.

**Final status.** FP16 for one FPN convolution passed the current offline performance, 32-image agreement, and full COCO PQ criteria. FP16 across all six encoder FFNs improved performance further and kept full PQ nearly flat, but only 26/32 images met the per-image gate. It remains an **experimental candidate awaiting acceptance**.

| Experiment | Key finding | Result and decision |
| --- | --- | --- |
| [Layer profiling](layer-profile.md) | The 12 encoder FFN GEMMs total 3.675 ms. The FPN 3×3 convolution is the most expensive single layer at about 0.903 ms | Validate the more localized FPN change first, then address the FFNs with the larger cumulative cost |
| [FP16 for one FPN convolution](fpn-half-ab.md) | Initial result 31/32. The anomaly followed logits when masks were swapped. Inspection found tactic changes in non-target backbone layers | Rebuilt the pair with a shared timing cache and reached 32/32. B1/S1 end-to-end P95 fell 2.6% |
| [FP16 across all six encoder FFNs](ffn-full-half.md) | 12 GEMMs switched to FP16 Tensor Cores. B1/S1 end-to-end P95 fell 13.7% | Full PQ was 53.18563 with 26/32 per-image passes. Retained as an experimental configuration |

## 1. Experiment Scope and Controls

Fix the RTX 4090, MaskDINO R50 panoptic, **1024×1024 / B1**, and the accepted base engine with model-level FP16 autocast / TF32 on. Its SHA-256 is `ab6054100b6256861ad320df440543d2e82cf2a833db2e77c159b736021401cf`. Some hotspots in this mixed-precision engine still use FP32 paths, and some GEMMs use TF32 Tensor Cores.

Stage 6 keeps the weight source, COCO data, FP32 engine I/O interface, and Stage 4–5 native C++/CUDA pre- and postprocessing and scheduling. The FPN experiment starts from the original mixed engine. The FFN experiment uses the **accepted FPN FP16 engine** as its control. The two percentage improvements use different baselines, so adding them directly would misstate the gain.

The measurements use three timing boundaries.

- **Layer and kernel profiling.** Preload GPU inputs and exclude JPEG handling, H2D/D2H, and panoptic postprocessing to locate bottlenecks. Layer profiling adds overhead, so **the sum of per-layer means differs from normal network latency**.
- **Network GPU time.** Measure the formal native pipeline with CUDA events and report P50/P95.
- **End-to-end latency and throughput.** Preload the same 500 JPEGs in memory and time from **dispatch to a slot** until the original-size CPU panoptic map and segment metadata are ready. Steady-state throughput uses the middle 90% of wall-clock time. Report B1/S1 and B1/S2 separately. This offline P95 excludes online queueing and does not measure online P99.

The formal A/B uses 20 warmup batches per slot and rotates engine and slot order across three rounds. NVML samples peak process GPU memory every 20 ms. Layer profiling, Nsight captures, and formal timing are separate. The environment uses driver 595.84, CUDA Toolkit 13.0.88, TensorRT 10.14.1.48, and reference container `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e`. [Environment and data](../docs/experiment-environment.md) · [Original build record](../stage3/results/precision/build-autocast-on/build.json)

**Correctness is evaluated at two levels.**

- **Per-image output agreement.** Compare a fixed set of 32 images with the reference engine. Require ≥99.9% pixel-level category agreement, a match for every segment, and same-class IoU ≥0.98 for each match. This measures output similarity on those images and leaves broader correctness unverified.
- **Full-set task accuracy.** Evaluate 5,000 COCO val2017 images against PyTorch FP32 B1 with the same inputs. Allow at most 0.05 points of loss **in each** of PQ, PQ Things, and PQ Stuff. This threshold defines acceptance for this experiment only.

[32-image and 500-image samples](../stage2/sample.json) · [PyTorch B1 PQ baseline](../stage3/results/pt-pq/result.json)

## 2. Hotspot Profiling and Selection of FPN and Encoder FFN

**Question.** After optimizing end-to-end execution and concurrency in Stages 4–5, what can still be optimized inside the base TensorRT engine?

**Method.** Profile the fixed B1 engine with `trtexec --separateProfileRun --dumpProfile`. Run one natively preprocessed COCO input three times, then use two other images to check hotspot ordering. Use Nsight Systems to correlate NVTX layer names, CUDA launches, and kernels. Collect resource counters for representative kernels in Nsight Compute.

| Layer / operator group | Layer time (ms/image) | Share of total layer time |
| --- | ---: | ---: |
| Encoder FFN (6 layers × 2 GEMMs) | **3.675** | **24.30%** |
| Encoder MSDeformAttn (6 plugin layers) | 1.480 | 9.78% |
| FPN `/sem_seg_head/layer_1/Conv` | **0.903** | **5.97%** |
| `input_proj.3` 3×3 Conv | 0.306 | 2.02% |
| Final mask Einsum + FP32 Cast | 0.171 | 1.13% |

Layer times sum to about **15.13 ms**, while independent normal timing gives network GPU P50 of about **14.09–14.11 ms**. Use the layer data to rank hotspots. End-to-end gains require separate measurements.

Kernel measurements show the following.

- **FPN 3×3 Conv.** Inputs and outputs are `[1,256,256,256]` FP32. The tactic uses FP32 Winograd. The representative kernel has **0%** Tensor pipe activity, uses no Tensor Cores, and reaches about **74.3%** of peak L2 throughput.
- **Representative encoder FFN linear2.** It already uses **TF32 Tensor Cores**. The representative kernel reaches about **67.8%** of peak DRAM throughput and **14.6%** of the active warp limit. A single set of resource counters guides investigation. A replacement must be benchmarked to establish speedup.

**Why start with FPN?** FFNs offer more cumulative optimization headroom, but they affect encoder features upstream of proposal/TopK, so numerical changes can propagate further. The target FPN convolution is the slowest individual layer, and its original ONNX data path feeds only the final `masks`. Start with the **more localized FPN change** as the first controlled experiment, then extend to FFNs.

**Result.** Profiling identified expensive compute regions and their execution paths. Whether FP16 is faster still requires builds and A/B measurements.

[Full layer profiling, Nsight data, and reproduction commands](layer-profile.md) · [Raw layer times](results/layer-profile-b1/summary.json) · [ONNX data path check](results/layer-profile-b1/onnx-paths.txt)

## 3. Experiment 1 · FP16 for One FPN 3×3 Convolution

### 3.1 Hypothesis and Implementation

**Hypothesis.** The target FPN convolution uses no Tensor Cores. Switching just this layer's computation to FP16 may lead TensorRT to choose an FP16 Tensor Core tactic and reduce total convolution and neighboring compute time even with added casts and reformats.

The [ONNX rewrite script](fpn_precision.py) matches `/sem_seg_head/layer_1/Conv` exactly, checks the original node, its exclusive weight initializer, and its output consumers, then makes the following changes.

1. Convert this convolution's FP32 weights to FP16.
2. Insert `Cast(FP32 → FP16)` before its input.
3. Cast the convolution output back to FP32 before the original GroupNorm branch.
4. Keep other ONNX nodes and engine I/O as they are. Rebuild the TensorRT engine and check the selected tactic in Engine Inspector.

Casting back to FP32 limits downstream precision changes. **FP16 rounding error already introduced persists.** Include explicit casts, reformats, and fused neighboring layers in the cost alongside convolution time.

### 3.2 Initial A/B · Faster Execution With 31/32 Passes

After the first independent builds of the control and FPN FP16 candidate, the target convolution dropped from about **0.909 ms to 0.251 ms**. The surrounding Add-to-GroupNorm/ReLU region dropped from about **1.283 ms to 0.819 ms**. Per-image alignment exposed an anomaly.

- Rebuilt control passed **32/32**.
- FPN FP16 candidate passed **31/32** at both S1 and S2.
- COCO image **470924** had **99.7614%** category agreement. The reference produced **72** segments and the candidate produced **71**.

The target convolution feeds only `masks` in ONNX. A mask change was the initial hypothesis for the missing segment and needed an explicit test.

### 3.3 Swapping Logits and Masks · The Anomaly Follows Logits

To localize the postprocessing difference, extract the same image's network outputs from the original reference, rebuilt control, and FP16 candidate. **Swap logits and masks across engines** while keeping the native postprocessing logic fixed.

**The FP16 candidate's logits produced 71 segments with every set of masks. The original reference and rebuilt control logits each preserved 72 segments with the candidate masks.**

This identifies **which network output triggered the postprocessing change**. The underlying kernel still needs to be isolated.

Engine Inspector showed that although ONNX changed one convolution in the FPN mask branch, **four backbone convolutions also changed tactics** in the first rebuild. The A/B therefore changed multiple execution paths. It cannot isolate the effect of the FPN convolution on logits.

**Result.** The 31/32 result was a real output anomaly. The experiment could not attribute it to FPN FP16 alone. Control non-target build differences and repeat the paired validation.

[Initial per-image validation](results/fpn-half-ab/alignment/) · [Initial tensor swap for 470924](results/fpn-half-ab/profile/470924-probe.json)

### 3.4 Rebuilding With a Shared Timing Cache

**Method.** Rebuild the control from the same original ONNX and use its TensorRT timing cache to build the FPN FP16 candidate. Compare tactics in Engine Inspector and recheck the anomalous image and all 32 samples.

The repeat produced the following results.

- Across 60 matching convolution and pooling layers, **only the target FPN convolution changed tactic**.
- For `470924`, the paired control and candidate **logits and boxes match elementwise**.
- The masks differ slightly (relative L2 **0.001342**). All nine logits × masks combinations retain **72** segments.
- The final candidate passes **32/32** at both S1 and S2.

**Inference and limits.** The repeat supports a link between the initial anomaly and rebuild differences outside the target path. Individual backbone tactics were never replaced one by one to establish a sole cause. Sharing the timing cache helps reduce incidental tactic differences. Verifying which execution details match still requires Engine Inspector and intermediate tensor checks.

**Changing one ONNX node can change several executed operators across TensorRT builds.**

[Shared-cache builds and tensor checks](results/fpn-half-ab/cached/) · [Full investigation and commands](fpn-half-ab.md)

### 3.5 Paired FPN Results and Acceptance

The **shared-cache rebuilt control** and FPN FP16 candidate were measured in the same session. Values are three-run medians.

| Metric | Control | FPN FP16 | Improvement |
| --- | ---: | ---: | ---: |
| Mean target convolution layer time | 0.9035 ms | 0.2505 ms | **−72.3%** |
| Adjacent Add → GroupNorm/ReLU | 1.2786 ms | 0.8203 ms | **−35.8%** |
| B1/S1 network GPU P95 | 14.17 ms | 13.76 ms | About −0.41 ms |
| B1/S1 end-to-end P95 | 16.27 ms | 15.85 ms | **−2.6%** |
| B1/S2 steady-state throughput | 74.21 images/s | 76.33 images/s | **+2.9%** |
| B1/S1 peak process GPU memory | 1.15 GiB | 1.15 GiB | No clear change observed |

![Paired native offline benchmark for the single-layer FPN optimization](results/fpn-half-ab/comparison.svg)

**Why did a 72% reduction in convolution time translate to a 2.6% end-to-end improvement?** The convolution is one part of the network and pre- and postprocessing. Casts, reformats, and unaffected operations still take time. A single-layer speedup cannot predict system-level speedup.

B1/S1 and B1/S2 each completed the full **5,000-image** COCO val2017 evaluation with unique and complete prediction IDs. Both scored **53.17180 / 59.01222 / 44.35606** for PQ / PQ Things / PQ Stuff, changes of **−0.01446 / −0.01504 / −0.01357 points** from PyTorch FP32 B1. Both pass the current **32/32** per-image gate.

**Decision. Accept the targeted FPN FP16 optimization** (acceptance commit `5f7b57f`). Acceptance covers the RTX 4090, B1/S1–S2, the specified offline samples, and full COCO PQ. Online P99 and sustained stability remain unverified.

[Per-run results, accuracy, and reproduction](fpn-half-ab.md) · [Raw A/B data](results/fpn-half-ab/cached/ab/summary.json) · [Full PQ](results/fpn-half-ab/pq/)

## 4. Experiment 2 · FP16 for 12 GEMMs Across Six Encoder FFNs

### 4.1 Why Optimize the FFNs Next?

FPN showed that a targeted precision change can improve performance, with a limited system-level gain. Profiling measured **3.675 ms (24.30%) across 12 encoder FFN GEMMs**, giving them more optimization headroom.

Each encoder FFN typically follows `256 → 2048 → 256` with two linear operations at about 21,760 spatial locations, for 12 GEMMs across six layers. FFNs affect encoder representations and downstream proposal and query paths, so numerical changes can propagate further than in the FPN experiment.

### 4.2 Graph Changes and Selected Kernels

Starting from the **accepted FPN FP16 ONNX**, the [FFN rewrite script](ffn_precision.py) keeps `linear1 → bias → ReLU → linear2 → bias` in FP16 for encoder layers 0–5, then casts back to FP32 for the original residual add and LayerNorm. It preserves the proposal TopK graph and the four FP32 engine I/O tensors.

The candidate build reuses a copy of the timing cache left by the paired FPN builds. Engine Inspector confirms that **all 12 GEMMs use FP16 Tensor Core tactics**. Some `linear2` operations fuse with conversions, residual operations, or normalization, so counting by TensorRT layer names alone misses them.

### 4.3 Paired Performance Results

This A/B uses **the engine with the accepted FPN optimization** as its control. The original Stage 3 mixed-precision engine was the starting point for the earlier FPN experiment.

| Metric | FPN control | FPN + six FP16 FFN layers | Improvement |
| --- | ---: | ---: | ---: |
| B1/S1 network GPU P95 | 14.01 ms | 11.78 ms | −2.23 ms |
| B1/S1 end-to-end P95 | 16.10 ms | 13.90 ms | **−13.7%** |
| B1/S1 steady-state throughput | 64.64 images/s | 75.41 images/s | +16.7% |
| B1/S2 end-to-end P95 | 27.87 ms | 24.28 ms | **−12.9%** |
| B1/S2 steady-state throughput | 76.39 images/s | 89.22 images/s | **+16.8%** |
| B1/S1 peak GPU memory | 1.15 GiB | 1.08 GiB | −0.07 GiB |
| B1/S2 peak GPU memory | 1.70 GiB | 1.56 GiB | −0.14 GiB |

![Offline results for FP16 across all six encoder FFNs](results/ffn-full-half/comparison.svg)

All three paired rounds improved in the same direction. **The performance hypothesis is supported. Acceptance also requires accuracy checks.**

#### Where the network time was saved

Reused the original FPN control (`d80102c6…`) and six-layer FFN FP16 candidate (`51e613b7…`) on the RTX 4090, from clean commit `301764c` and the same reference container. Nsight Systems 2025.5.1.121 captured CUDA and NVTX at 1024×1024, B1/S1, with the same natively preprocessed COCO 149770 input and benchmark I/O transfers disabled. Each plan warmed up for 500 ms and ran 40 measured inferences. The capture included warm-up, so the analysis used the last 40 `enqueueV3` ranges per plan. [Reference environment](ffn-full-half.md#measurement-definitions)

For each inference, match Engine Inspector layers to NVTX ranges using ONNX metadata, match CUDA launches by thread and range, then join launches to GPU kernels by correlation ID. The comparison covers all six blocks, from `norm1` through `linear2` and the output boundary. That boundary includes the next attention input add for blocks 0–4 and the encoder output slice/transpose for block 5 because these operations fuse differently across the plans. The analysis verified complete coverage of each block on a single stream.

| GPU kernel work across six blocks | FPN control (ms) | FFN FP16 (ms) | Time saved (ms) |
| --- | ---: | ---: | ---: |
| `norm1` and FFN input cast | 0.230 | 0.287 | −0.057 |
| `linear1` with bias and ReLU | 1.740 | 0.719 | **1.021** |
| `linear2` through the output boundary, including residual and `norm2` | 2.442 | 1.311 | **1.131** |
| **FFN and boundaries** | **4.411** | **2.317** | **2.094** |
| Other network kernels | 8.850 | 8.852 | −0.002 |
| **All network kernels** | **13.261** | **11.169** | **2.092** |

The trace attributes essentially all of its kernel-time reduction to the FFN and its boundaries. Both GEMMs already used TF32 Tensor Cores in the control. The candidate uses FP16 Tensor Core tactics, with `linear2` fused with the FP32 conversion, residual add and `norm2`. The trace measures the combined gain from the new tactic and fusion. Quantifying their individual contributions or changes in DRAM traffic would require additional controls or hardware counters.

These values are mean sums of kernel durations on one input. The original 500-image native benchmark measured the **2.23 ms network P95 reduction** with CUDA events. The separate `trtexec` layer exports cross-check the attribution: the same FFN boundaries save **2.123 ms** out of **2.122 ms** in summed mean layer time, with the remaining layers effectively flat. [Control layer export](results/ffn-full-half/ab/fpn.layers.json) · [Candidate layer export](results/ffn-full-half/ab/ffn-half.layers.json)

For each plan, mount the original engine at `/network.engine` and the input tensor at `/input.f32`. Run both `nsys` commands in a writable bind-mounted directory inside the reference container with a distinct `label` per plan. [Input checksums](results/layer-profile-b1/inputs.sha256)

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output="$label" \
  /opt/tensorrt/bin/trtexec --loadEngine=/network.engine --loadInputs=image:/input.f32 \
  --infStreams=1 --noDataTransfers --warmUp=500 --duration=0 --iterations=40
nsys export --type sqlite --output "$label.sqlite" "$label.nsys-rep"
```

From the repository root, point `trace_dir` to the exported SQLite files and run the summary with the Inspector export for each plan.

```bash
python3 stage6/ffn_trace_summary.py "$trace_dir/fpn.sqlite" \
  stage6/results/fpn-half-ab/cached/fpn-half/layers.json "$trace_dir/fpn.csv"
python3 stage6/ffn_trace_summary.py "$trace_dir/ffn-half.sqlite" \
  stage6/results/ffn-full-half/build/layers.json "$trace_dir/ffn-half.csv"
```

Raw reports and SQLite exports are on the 4090 at `/home/parker/Code/MaskDINO-TensorRT-layer-profile-20261110/repo/stage3/runs/ffn-attribution-x8sP2E/`. [Report checksums](results/ffn-full-half/attribution/reports.sha256) · [Control per-inference timings](results/ffn-full-half/attribution/fpn.csv) · [Candidate per-inference timings](results/ffn-full-half/attribution/ffn-half.csv) · [Control kernel mapping](results/ffn-full-half/attribution/fpn.kernels.csv) · [Candidate kernel mapping](results/ffn-full-half/attribution/ffn-half.kernels.csv)

### 4.4 Nearly Flat Full PQ With Only 26/32 Per-Image Passes

Both slot settings evaluated all **5,000 COCO images** and produced identical results.

| Metric | Accepted FPN | FPN + six FP16 FFN layers |
| --- | ---: | ---: |
| PQ | 53.17180 | **53.18563** |
| PQ Things | 59.01222 | **59.02411** |
| PQ Stuff | 44.35606 | **44.37283** |
| 32-image per-image gate | 32/32 | **26/32** |

The FFN candidate changes PQ / PQ Things / PQ Stuff by only **−0.00062 / −0.00315 / +0.00319 points** from PyTorch FP32 B1 and passes the project's full-set accuracy thresholds. Six images fail the per-image check. **COCO 435081** has only **85.6848%** pixel-level category agreement, a minimum same-class IoU of **0.16608**, and one unmatched segment on each side.

**PQ measures overall task accuracy against ground truth. Per-image agreement measures changes in specific outputs against the control.** Models with similar PQ can differ substantially on individual images. Report both metrics so that known per-image differences remain visible.

**Observed.** The 12 FP16 GEMMs yield reproducible performance gains, with substantial output changes on some images.

**Open question.** Whether encoder outputs, proposal scores, TopK cutoff, or decoder query selection amplify numerical error remains unverified. TopK is a plausible lead. The current record lacks causal tests of intermediate features, candidate rankings, and counterfactual inputs.

**Decision. Retain as an experimental configuration.** Direct replacement under strict per-image agreement criteria is unsupported. An application that tolerates output differences and only requires task accuracy needs its own metrics and independent acceptance set before using these COCO results for a decision.

[Full FFN report, six failing images, and reproduction](ffn-full-half.md) · [Per-run results](results/ffn-full-half/ab/summary.json) · [Full evaluation and performance summary](results/ffn-full-half/summary.json)

## 5. Findings and Limits

| Question | Verified finding | Unverified claim |
| --- | --- | --- |
| Can profiling guide optimization? | Identified the slowest single FPN layer and the largest cumulative hotspot in encoder FFN. Used kernel evidence to design A/B tests | A single Nsight utilization figure can predict speedup |
| Is targeted FPN FP16 effective? | Target convolution and native end-to-end performance both improve and pass the current project accuracy gates | FP16 is risk-free across data and hardware |
| How can attribution errors be avoided? | The original anomaly followed logits and non-target backbone tactics changed. The anomaly disappeared with the shared-cache pair | A shared cache fixes all engine execution details, or the sole causal kernel has been identified |
| Does FFN FP16 help? | Latency drops and throughput improves substantially. Full COCO PQ is nearly flat | Per-image outputs remain equivalent, or TopK has been established as the root cause |
| Do the tests represent production? | Reproducible on the specified 4090, 500 offline images, and COCO val2017 | Online queueing, online P99, stability across GPUs, and real application data have passed acceptance |

**Follow-up priorities.** Test FP16 sensitivity by encoder FFN layer or group. Inspect encoder features, TopK indices, and scores near the cutoff for failing images. If needed, keep sensitive FFNs in FP32 and compare performance and restored per-image agreement. Further acceptance should cover independent holdout images, online Poisson arrivals, online P99, and sustained stability.

## 6. Reproduction and Experiment Archives

All figures come from archived RTX 4090 experiments. This report summarizes those records and adds no GPU rerun results. Before reproducing, verify GPU availability, original engines and ONNX, the reference container, weights, and sample SHA-256 hashes. Original runs used different commits and historical paths may differ from the current `stage6/` entry points. Use the reproduction commands in each detailed report.

| Experiment | Original commit / acceptance | Code and procedure | Results |
| --- | --- | --- | --- |
| Layer timing and Nsight | `8e3964b` | [layer-profile.md](layer-profile.md#reproduction) · [profile_layers.sh](profile_layers.sh) | [layer-profile-b1](results/layer-profile-b1/) |
| Initial FPN build, per-image anomaly, and tensor swapping | `672f0aa`, `d9e30d7`, `9237433` | [fpn-half-ab.md](fpn-half-ab.md#reproduction) · [fpn_probe.py](fpn_probe.py) | [Initial A/B](results/fpn-half-ab/alignment/) · [Tensor recheck](results/fpn-half-ab/profile/470924-probe.json) |
| Shared FPN cache, formal A/B, and full PQ | `54e6f15`, accepted at `5f7b57f` | [fpn_precision.py](fpn_precision.py) · [fpn_ab.py](fpn_ab.py) | [Paired records](results/fpn-half-ab/cached/) · [Full PQ](results/fpn-half-ab/pq/) |
| Encoder FFN FP16 experiment | `301764c` | [ffn_precision.py](ffn_precision.py) · [ffn_ab.py](ffn_ab.py) | [ffn-full-half](results/ffn-full-half/) |

Historical batch × slot and online workload experiments are still available under `stage6/` at commit `fcc7dc9`. FPN and FFN acceptance here covers the experiments above.
