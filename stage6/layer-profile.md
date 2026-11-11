# Layer Profiling of the FP16 Mixed-Precision Engine

**English** · [简体中文](layer-profile.zh-CN.md)

Fix the RTX 4090, MaskDINO R50 panoptic, 1024×1024, and B1/S1. Load the model-level FP16 autocast / TF32-on plan accepted for Stage 6 directly. Engine SHA-256 is `ab6054100b6256861ad320df440543d2e82cf2a833db2e77c159b736021401cf`. The original build ran at `81a7d618315fa6df2702aba8ea4a22b77d72620a` and this profiling run at `8e3964b29c0f082b7d2ade365cf6241ebc2e55f0`. The environment uses driver 595.84, CUDA Toolkit 13.0.88, TensorRT 10.14.1.48, and image `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e`. Reuse the COCO annotations, weights, and ONNX source from the existing build record. [Build details](../stage3/results/precision/build-autocast-on/build.json) · [Build log](../stage3/results/precision/build-autocast-on/console.log) · [Run log](results/layer-profile-b1/console.log)

## Measurement Method

- Eight consecutive one-second samples before launch showed 0% GPU utilization, 0% memory controller utilization, and 110 MiB, with only Xorg and GNOME Shell. One-second sampling during layer profiling recorded the preprocessing process and five `trtexec` processes. The experiment uses default power and clock policies, and sampling can detect sustained utilization. [GPU samples](results/layer-profile-b1/gpu-samples.csv) · [Compute process samples](results/layer-profile-b1/compute-processes.csv)
- Select COCO 149770 from the 32-image list and run it three times in separate processes. Run 456394 and 107339 once each to check rankings. Stage 4 native preprocessing generates FP32 NCHW input, which is preloaded to the GPU. Disable H2D/D2H in the timing loop and exclude JPEG preprocessing and panoptic postprocessing. Each run warms up for 2 seconds and runs normal timing for at least 5 seconds and 300 iterations. Actual counts are 356–357. [Input checksums](results/layer-profile-b1/inputs.sha256)
- `--separateProfileRun` collects layer timings separately. The profiler export count is 419 per run. The table takes the median of the three runs for 149770. Within each group, sum the corresponding layer `averageMs` per run and divide by the sum of all layer means in that run for the share. The sum of layer means is about 15.13 ms. In normal timing, GPU Compute Time P50 is 14.086–14.105 ms and P95 is 14.100–14.121 ms. The two measurements run separately. Layer time is used to rank hotspots. [Per-run timings and grouped results](results/layer-profile-b1/summary.json) · [Raw layer export](results/layer-profile-b1/149770-r1.layers.json)

## Hotspots

| Layer or group | Time (ms/image) | Share of layer time |
| --- | ---: | ---: |
| 12 encoder FFN GEMMs | **3.675** | **24.30%** |
| 6 encoder MSDeformAttn plugin layers | 1.480 | 9.78% |
| `/sem_seg_head/layer_1/Conv`, most expensive single layer | **0.903** | **5.97%** |
| `/sem_seg_head/input_proj.3/input_proj.3.0/Conv` | 0.306 | 2.02% |
| Final mask Einsum and output FP32 Cast | 0.171 | 1.13% |
| 9 transformer decoder MSDeformAttn plugin layers | 0.093 | 0.61% |

`layer_1/Conv` is the most expensive single layer in all five runs, ranging from 0.901–0.905 ms. The six encoder FFN `linear2` GEMMs rank 2–7 among individual layers. The six `linear1` GEMMs total about 1.761 ms. FFN time on the two additional inputs totals 3.675 and 3.678 ms. [Per-run data](results/layer-profile-b1/summary.json) · [Engine Inspector](../stage3/results/precision/build-autocast-on/layers.json)

## Kernel Analysis

In a separate capture, Nsight Systems 2025.5.1 correlates NVTX layer names with CPU launches, then uses CUDA correlation IDs to link GPU kernels. Each NVTX layer below has 40 observed kernel instances. Nsight Compute 2025.3.1 profiles each kernel type once with SpeedOfLight and LaunchStats and leaves clocks and cache behavior at their natural settings. Counters characterize each kernel. The table's Nsight Systems times are medians for the corresponding kernels in the trace. [Raw Nsight Systems report](results/layer-profile-b1/layer-kernels.nsys-rep) · [Nsight Compute method](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay)

| Representative kernel | Nsight Systems (µs) | Tensor pipe activity | L2 throughput as a share of peak | DRAM throughput as a share of peak | Active warps as a share of the limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| FPN FP32 Winograd 3×3 | 910 | 0.0% | 74.3% | 15.0% | 32.8% |
| Encoder layer 1 `linear2`, zero-based | 324 | 41.8% | 45.3% | 67.8% | 14.6% |
| Encoder MSDeformAttn | 256 | 0.0% | 38.9% | 26.5% | 91.7% |

`layer_1/Conv` has FP32 `[1,256,256,256]` inputs and outputs and uses a Winograd tactic. This kernel has high L2 activity and uses no Tensor Cores. The representative FFN `linear2` already uses TF32 Tensor Cores and 238 registers per thread. These single-kernel counter samples guide the next A/B. Utilization alone cannot establish the gain from switching tactics. [FPN counters](results/layer-profile-b1/conv-hot.ncu.csv) · [FFN counters](results/layer-profile-b1/encoder-ffn-linear2.ncu.csv) · [MSDeformAttn counters](results/layer-profile-b1/msda.ncu.csv)

## Next Optimization

Prioritize a **single-layer precision A/B on `/sem_seg_head/layer_1/Conv`**. In the original ONNX graph, this convolution feeds only `masks`. Encoder FFNs feed TopK and all three engine outputs. First convert convolution inputs and weights to FP16 and cast the output back to FP32 before the original GroupNorm. Check for a Tensor Core tactic and compare the total time of the convolution and new casts or reformats. If explicit conversions offset the gain, evaluate adjacent layouts and precision boundaries. [ONNX data path check](results/layer-profile-b1/onnx-paths.txt) · [Upstream FPN implementation](https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/modeling/pixel_decoder/maskdino_encoder.py#L382-L397)

Encoder FFNs offer more cumulative optimization headroom and come second. Select only a few FFN layers at a time, keep residuals, normalization, and TopK in FP32, and check whether proposals change. Check candidate outputs on 32 images. Use the same 500 JPEGs to compare B1/S1 latency, B1/S2 saturated throughput, and GPU memory. Evaluate the final candidate on all of COCO with the Stage 6 limit of a 0.05-point drop in each PQ metric. This section records the choice at profiling time. See the [single-layer FPN experiment](fpn-half-ab.md) for the implemented change and A/B results.

## Reproduction

Use a clean checkout on the 4090 and set the accepted engine path. The script checks for existing CUDA compute processes and saves GPU samples during layer profiling. Use a new run ID each time.

```bash
export MASKDINO_IMAGE=sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e
data=/path/to/coco2017-panoptic-val
engine=/path/to/accepted/fp16-b1/network.engine
bash stage6/profile_layers.sh "$data" "$engine" layer-profile-b1
```

The complete Nsight commands follow. Use `149770.f32` from the timing script to keep the capture input fixed. On this host, Nsight Compute reads performance counters with a temporary container capability. [NVIDIA performance counter permissions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)

```bash
out="$PWD/stage6/runs/layer-profile-b1"
container=(--rm --gpus all --network=none -e HOME=/tmp
  --mount "type=bind,source=$engine,target=/network.engine,readonly"
  --mount "type=bind,source=$out,target=/run-output" -w /run-output)
infer=(/opt/tensorrt/bin/trtexec --loadEngine=/network.engine
  --loadInputs=image:149770.f32 --infStreams=1 --noDataTransfers)

docker run "${container[@]}" --user "$(id -u):$(id -g)" \
  --entrypoint nsys "$MASKDINO_IMAGE" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output=layer-kernels \
  "${infer[@]}" --warmUp=500 --duration=0 --iterations=8

profile_kernel() {
  local name=$1 kernel=$2 skip=$3
  docker run "${container[@]}" --cap-add=SYS_ADMIN --entrypoint ncu "$MASKDINO_IMAGE" \
    --clock-control none --cache-control none --kernel-name-base function \
    --kernel-name "regex:$kernel" --launch-skip "$skip" --launch-count 1 \
    --section SpeedOfLight --section LaunchStats --export "$name" \
    "${infer[@]}" --warmUp=0 --duration=0 --iterations=20
}

profile_kernel conv-hot trt_ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1 8
profile_kernel encoder-ffn-linear2 sm86_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x16_stage3_warpsize2x2x1_tensor16x8x8_execute_kernel_trt 321
profile_kernel msda ms_deformable_im2col_gpu_kernel 30
```

The FFN `--launch-skip` is based on the order of identically named kernels in this Nsight Systems trace. Multiple layers use the symbol. After rebuilding the plan, locate the kernel again through NVTX. Raw reports, SQLite exports, and inputs are on the 4090 at `/home/parker/Code/MaskDINO-TensorRT-layer-profile-20261110/repo/stage3/runs/layer-profile-b1/`. [Report checksums](results/layer-profile-b1/reports.sha256)
