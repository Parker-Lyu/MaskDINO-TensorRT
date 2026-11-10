# FP16 混合 engine 的逐层 profiling

固定 RTX 4090、MaskDINO R50 panoptic、1024×1024、B1/S1。直接加载 Stage 6 验收的模型级 FP16 autocast / TF32 on plan。Engine SHA-256 为 `ab6054100b6256861ad320df440543d2e82cf2a833db2e77c159b736021401cf`。原构建提交为 `81a7d618315fa6df2702aba8ea4a22b77d72620a`，本次采样提交为 `8e3964b29c0f082b7d2ade365cf6241ebc2e55f0`。环境为驱动 595.84、CUDA Toolkit 13.0.88、TensorRT 10.14.1.48，镜像为 `sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e`。COCO 标注、权重与 ONNX 来源沿用现有构建记录。 [构建信息](results/precision/build-autocast-on/build.json) · [构建日志](results/precision/build-autocast-on/console.log) · [运行日志](results/layer-profile-b1/console.log)

## 测量方法

- 启动前连续 8 次每秒采样为 0% GPU 和 0% 显存控制器利用率，110 MiB，只有 Xorg 和 GNOME Shell。逐层实验每秒采样，记录到本次预处理和 5 个 `trtexec` 进程。使用默认功耗及频率策略，采样可以发现持续占用。 [GPU 采样](results/layer-profile-b1/gpu-samples.csv) · [计算进程采样](results/layer-profile-b1/compute-processes.csv)
- 从 32 张清单选 COCO 149770，重复三轮，每轮新建进程；再分别用 456394、107339 各跑一轮检查排序。Stage 4 原生前处理生成 FP32 NCHW 输入，预加载到 GPU。关闭计时循环内的 H2D/D2H，JPEG 前处理和 panoptic 后处理也在计时范围外。每轮预热 2 秒，普通计时至少 5 秒、至少 300 次，实际 356–357 次。 [输入校验](results/layer-profile-b1/inputs.sha256)
- `--separateProfileRun` 单独采集层耗时，每轮 profiler 导出计数为 419。下表取 149770 三轮的中位数。组内先按每轮将对应层的 `averageMs` 相加，占比以同一轮全部层的均值之和为分母。逐层均值总和约 15.13 ms，普通计时的 GPU Compute Time P50 为 14.086–14.105 ms，P95 为 14.100–14.121 ms。两种计时分开执行，层耗时适用于比较热点排序。 [逐轮计时与分组结果](results/layer-profile-b1/summary.json) · [逐层原始导出](results/layer-profile-b1/149770-r1.layers.json)

## 热点

| 层或组 | 耗时（ms/图） | 逐层时间占比 |
| --- | ---: | ---: |
| Encoder FFN 的 12 个 GEMM | **3.675** | **24.30%** |
| Encoder MSDeformAttn 的 6 个插件层 | 1.480 | 9.78% |
| `/sem_seg_head/layer_1/Conv`，最贵的单层 | **0.903** | **5.97%** |
| `/sem_seg_head/input_proj.3/input_proj.3.0/Conv` | 0.306 | 2.02% |
| 最后一次 mask Einsum 和输出 FP32 Cast | 0.171 | 1.13% |
| Transformer decoder 的 9 个 MSDeformAttn 插件层 | 0.093 | 0.61% |

五轮的最贵单层均为 `layer_1/Conv`，范围 0.901–0.905 ms。六个 encoder FFN `linear2` GEMM 占据单层第 2–7 名，六个 `linear1` GEMM 合计约 1.761 ms。两个额外输入的 FFN 总耗时为 3.675 和 3.678 ms。 [逐轮数据](results/layer-profile-b1/summary.json) · [Engine Inspector](results/precision/build-autocast-on/layers.json)

## Kernel 复核

Nsight Systems 2025.5.1 在独立采集中通过 NVTX 层名关联 CPU launch，再按 CUDA correlation ID 关联 GPU kernel。每个下列 NVTX 层观测到 40 次 kernel。Nsight Compute 2025.3.1 对每种 kernel 单独采一次，使用 SpeedOfLight 和 LaunchStats，使用自然频率和 cache 行为。计数器用来解释 kernel，表中的 Nsight Systems 时间是 trace 中对应 kernel 的中位数。 [Nsight Systems 原始报告](results/layer-profile-b1/layer-kernels.nsys-rep) · [Nsight Compute 方法](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay)

| 代表 kernel | Nsight Systems（µs） | Tensor pipe 活跃 | L2 吞吐占峰值 | DRAM 吞吐占峰值 | 活跃 warp 占上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FPN FP32 Winograd 3×3 | 910 | 0.0% | 74.3% | 15.0% | 32.8% |
| Encoder 第 1 层 `linear2`，零起始 | 324 | 41.8% | 45.3% | 67.8% | 14.6% |
| Encoder MSDeformAttn | 256 | 0.0% | 38.9% | 26.5% | 91.7% |

`layer_1/Conv` 的输入和输出均为 `[1,256,256,256]` FP32，使用 Winograd tactic。该 kernel 的 L2 活动较高且未使用 Tensor Core。代表性 FFN `linear2` 已使用 TF32 Tensor Core，单线程使用 238 个寄存器。以上是单次 kernel 计数器采样，足以指导下一组 A/B，资源利用率本身无法证明更换 tactic 的收益。 [FPN counters](results/layer-profile-b1/conv-hot.ncu.csv) · [FFN counters](results/layer-profile-b1/encoder-ffn-linear2.ncu.csv) · [MSDeformAttn counters](results/layer-profile-b1/msda.ncu.csv)

## 下一处改动

优先对 **`/sem_seg_head/layer_1/Conv` 做单层精度 A/B**。原 ONNX 数据流显示该卷积只通向 `masks`，encoder FFN 则通向 TopK 和三个 engine 输出。先把卷积输入与权重转成 FP16，卷积输出在原 GroupNorm 前转回 FP32。检查是否选到 Tensor Core tactic，并比较卷积和新增 Cast 或 reformat 的总耗时。若显式转换抵消收益，再评估相邻布局和精度边界。 [ONNX 路径核对](results/layer-profile-b1/onnx-paths.txt) · [上游 FPN 实现](https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/modeling/pixel_decoder/maskdino_encoder.py#L382-L397)

Encoder FFN 的累计优化空间更大，第二步再做。一次只选少量 FFN 层，保留残差、归一化和 TopK 的 FP32 表示，检查是否出现 proposal 翻转。候选用 32 张图检查输出和差异，用同一份 500 张 JPEG 比较 B1/S1 延迟、B1/S2 饱和吞吐和显存，最终候选按 Stage 6 的三项 PQ 各最多下降 0.05 点进行全量 COCO 验收。本轮以 profiling 定位改动点，表中没有候选 engine 的加速数据。

## 复现

在 4090 上使用干净 checkout，设置验收过的 engine 路径。脚本会检查是否已有 CUDA 计算进程，并在本次逐层实验期间保存 GPU 采样。每次使用新运行 ID。

```bash
export MASKDINO_IMAGE=sha256:24a707391a7592235352cb91d58e11da356aefe3f7512a698a5402671cd83d1e
data=/path/to/coco2017-panoptic-val
engine=/path/to/accepted/fp16-b1/network.engine
bash stage3/profile_layers.sh "$data" "$engine" layer-profile-b1
```

Nsight 的完整命令如下。使用计时脚本生成的 `149770.f32`，避免采集时更换输入。Nsight Compute 在该主机使用临时容器能力读取性能计数器。 [NVIDIA 性能计数器权限说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)

```bash
out="$PWD/stage3/runs/layer-profile-b1"
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

FFN 的 `--launch-skip` 由本次 Nsight Systems trace 的同名 kernel 顺序确定。该符号被多个层复用，重建 plan 后要重新通过 NVTX 关联定位。原始报告、SQLite 导出和输入在 4090 的 `/home/parker/Code/MaskDINO-TensorRT-layer-profile-20261110/repo/stage3/runs/layer-profile-b1/`。 [报告校验](results/layer-profile-b1/reports.sha256)
