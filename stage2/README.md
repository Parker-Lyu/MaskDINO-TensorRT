# Stage 2 · Adapt Preprocessing and Establish a Performance Baseline

**English** · [简体中文](README.zh-CN.md)

This stage fixes the MaskDINO R50 panoptic weights to establish the PyTorch reference before TensorRT conversion. Standardize the input shape and fix two sample sets, then use full PQ to select an input size and offline benchmarks to quantify speed and GPU memory costs. The resulting baseline is **1024×1024**, with **B1** as the priority for the next stage. [Stage decision, full results tables, and comparison chart](../README.md#stage-2--adapt-preprocessing-and-establish-a-performance-baseline)

## Inputs and Outputs

To batch images at a common size, decode JPEGs to RGB with EXIF orientation applied and resize proportionally to the target long side using Detectron2 bilinear resize. Round the short side with Python `round` and clamp it to at least 1. Pad the right and bottom with 128 in pixel space, matching the default for `FixedSizeCrop` in the official R50 panoptic training mapper, then normalize using the official configuration. Bilinearly interpolate mask logits to the square canvas, crop to the valid region, restore them to the original image size, and run the official panoptic merge. Inference uses FP32 with autocast, TF32, and OOM fallback disabled.

Stage 1 uses the official input policy of a short side of 800 and a maximum long side of 1333. The PQ difference between stages reflects the entire input policy change, including resizing and padding.

## Data and Validation Uses

| Data | Selection method | Use |
| --- | --- | --- |
| 32 images for numerical validation | Cover extremes in original image area, aspect ratio, Things count, and panoptic segment count, then fill the sample with a fixed seed | Check synthetic geometry for landscape, portrait, and square images first, then compare B1 and B2/B4/B8 logits, ID maps, and segment metadata per image |
| 500 images for performance validation | Stratify by original pixel-count tertiles and aspect ratio thresholds of 0.9 and 1.1, allocate proportionally to the full set, and use seed 20261106 | Fix image IDs and execution order to compare input sizes and batches on the same workload |
| All 5,000 COCO 2017 validation images | Use the official images and panoptic annotations | Run B4 PQ once at each of the three resolutions and verify full prediction coverage |

[sample.json](sample.json) stores the two sample sets as `smoke_image_ids` and `image_ids`. The 500-image sample averages 11.47 segments per image, versus 11.35 for the full set. Non-crowd Things smaller than 32² pixels account for 43.47% and 42.37%, respectively. Lighting was excluded from stratification.

## Reproduction

### Environment and Run Revisions

The reference environment uses an RTX 4090, driver 595.84, CUDA 13.0, and PyTorch `2.10.0a0+b558c986e8.nv25.11`. Reuse the [Stage 1 image](../stage1/README.md) and [data and weights](../docs/experiment-environment.md). The official runs used image ID `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e`. The launcher verifies the annotation and checkpoint SHA-256 hashes, requires a clean working tree, and writes outputs to `stage2/runs/<run-id>/`.

| Experiment | Run commit |
| --- | --- |
| 32-image numerical validation, B1/B2/B4/B8 | `7b0decc6145fbac5508b8f8e6cb0cdc0173ee324` |
| 640/800/1024 full PQ, B1 rerun, and B1/B4 performance comparison at 800/1024 | `7d8788b99327dd359b2cad29d8ae14141660d8da` |
| Follow-up 1024 B2/B8 benchmarks | `1f96771234baefb607dceb61092fa6474fd17123` |
| Results archive, plotting data, and aggregation scripts | `stage2-pytorch-square` |

Run the following commands from the repository root, then run the experiment blocks in order in the same Bash session. Set absolute paths to existing data and weights. Add a timestamp to `MASKDINO_RUN_ID` to avoid collisions with existing result directories.

```bash
repo=$(git rev-parse --show-toplevel)
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
git worktree add --detach "${repo}-stage2-repro-${run_stamp}" stage2-pytorch-square
cd "${repo}-stage2-repro-${run_stamp}"

data=/absolute/path/to/coco2017-panoptic-val
export MASKDINO_CHECKPOINT=/absolute/path/to/maskdino_r50_panoptic.pth
export MASKDINO_IMAGE=maskdino:stage1
```

### 1. Sampling and Numerical Validation

Run B1 first at each size to save reference outputs for that size. Reuse the same run ID within each batch-size comparison.

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

### 2. Full PQ at Three Input Sizes

Evaluation uses Detectron2's `COCOPanopticEvaluator` and the official validation set category mapping. The evaluation loop includes reading JPEGs from disk and encoding evaluator PNGs. Use the separate entry point in the next section for performance comparisons.

```bash
git switch --detach 7d8788b99327dd359b2cad29d8ae14141660d8da
export MASKDINO_RUN_ID="pq-${run_stamp}"
for size in 640 800 1024; do
  bash stage2/run.sh "$data" evaluate "$size" 4
done
```

### 3. B1/B4 Performance at 800 and 1024

Run each configuration in a separate container. Preload the JPEGs, use the first input batch to warm up for 20 batches, then measure all 500 images. Rotate the configuration order across the three runs.

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

### 4. Follow-Up B2/B8 Benchmarks at 1024

The follow-up commit records cumulative wall-clock time at each batch end and uses the same inference path. The 500-image B8 run contains 62 full batches and a final batch of four images. Use the first 496 images for a separate full-batch B2/B8 comparison.

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

### 5. Aggregation

The aggregation script is in the acceptance tag. Check out that tag to summarize the runs. Outputs in `stage2/runs/` remain available.

```bash
git switch --detach stage2-pytorch-square
python3 stage2/summarize_benchmark.py stage2/runs/perf-"${run_stamp}"-r*/benchmark/*/result.json
python3 stage2/summarize_benchmark.py stage2/runs/batch-"${run_stamp}"-r*/benchmark/*/result.json
python3 stage2/summarize_benchmark.py --prefix-images 496 \
  stage2/runs/batch-"${run_stamp}"-r*/benchmark/*/result.json
```

## Metric Definitions

- **Network GPU latency** uses CUDA events around the backbone and segmentation head. P50 and P95 are per batch.
- **End-to-end latency** runs from the start of processing the batch's in-memory JPEGs until the last image's CPU panoptic map and segment metadata are ready. P50 and P95 are per batch. In the summary JSON, `batch_e2e_*` contains batch metrics and `e2e_*` retains per-request metrics. The offline workload uses preassembled batches. Timing includes decoding, preprocessing, H2D, inference, postprocessing, and D2H.
- **Offline end-to-end throughput** divides the number of completed images by full-run wall-clock seconds, including CPU scheduling overhead inside the loop. Preloading and warmup finish before measurement. The full-batch comparison uses the cumulative time when image 496 completes.
- **Peak GPU memory** uses PyTorch CUDA allocator peak allocated and peak reserved after resetting the peaks following warmup. The measurement covers the full run.
- Tables report medians of each metric across three runs. Per-run minima, maxima, and raw samples are in the results directories.

## Validation Results and Archives

- **Numerical validation.** All 12 probes completed. Segment metadata for the 32 images matched across configurations. The worst per-image B4 vs. B1 ID-map pixel difference was 0 at 640 and 800, and 0.000326% at 1024, so all full evaluations used B4. The largest absolute differences in B4 per-query mask logits were 52.93 at 800 and 152.44 at 1024. Queries were not rematched for this comparison, and the cause remains unverified. The B1 rerun matched logits, ID maps, and metadata exactly. [Per-image records and B1 rerun](results/probe/)
- **Full PQ.** Each resolution produced 5,000 unique prediction IDs, exactly matching ground truth. PQ was 49.9662, 52.0120, and 53.1862 at 640, 800, and 1024, respectively. [Evaluation logs and metric JSON](results/evaluate/)
- **Performance at 800 and 1024.** All 12 runs completed 500 images with zero OOMs or fallbacks. End-to-end P95 was 51.93 ms for 1024/B1 and 37.85 ms for 800/B1. The 1.1742-point PQ advantage led to selecting 1024 as the deployment baseline to preserve accuracy. [Per-run statistics and raw data](results/benchmark/)
- **Follow-up B2 and B8 benchmarks at 1024.** All six runs completed 500 images with zero OOMs or fallbacks. On the full 500-image basis used in the root README, throughput was 21.33 and 22.56 images/s, with batch end-to-end P95 of 97.81 and 365.04 ms, respectively. For the first 496 images in full batches, throughput was 21.33 and 22.57 images/s, with P95 of 97.83 and 365.06 ms. B8's final partial batch spent 138.90–141.55 ms in the network. Its last CPU output was ready 183.28–186.11 ms after batch start. [500-image summary](results/batch-sweep/summary-500.json) · [496-image full-batch summary](results/batch-sweep/summary-496.json) · [Batch details and logs](results/batch-sweep/)

Complete run directories are on the 4090 host at `/home/parker/Code/MaskDINO-TensorRT-stage2/stage2/runs/`. The repository archives metrics, batch details, and evaluation logs. Raw prediction JSON and large B1 reference tensors remain on the experiment host.
