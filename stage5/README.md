# Stage 5 · 用多 CUDA stream 实现异步并发

固定 RTX 4090、MaskDINO R50 panoptic、1024×1024、TensorRT FP32、关闭 TF32，复用 Stage 4 的 B1 engine 和 C++/CUDA 前后处理。每个请求处理一张内存 JPEG，交付原图尺寸的 CPU panoptic map 和 segment metadata。固定 batch 为 1，只扫 slot 数。批大小与在线负载的比较见 [Stage 6](../stage6/README.md)。

## Slot 生命周期

同一进程共享一份反序列化的 engine。每个 slot 对应一个长期运行的 CPU worker，独立持有 CUDA stream、TensorRT execution context、可写 I/O、前后处理工作区和 pinned staging buffers。worker 在本 slot 的 stream 上提交前处理、`enqueueV3`、后处理与 D2H，等待最终 D2H 后把 map 和 metadata 拷贝到 CPU 结果对象，然后复用 slot。等待期间，其他 worker 可以继续提交或收取结果。运行时要求 engine 的辅助 stream 数为 0，图中的并发来自多个请求的独立 slot。 [调度器](main.cu) · [原生运行时](../stage4/runtime.cu)

## Nsight Systems

用 `c8d4796` 在同一份 32 张 COCO 清单上独立采集，每 slot 预热 20 批。图中两组都从各自第一条 GPU kernel 开始，展示前 60 ms。蓝色按 CUDA correlation ID 关联到同一 CPU 线程的 `TensorRT inference` NVTX 区间，灰色为其余 kernel。橙色标出至少两个 stream 同时执行 TensorRT kernel 的区间。Req 编号按每份 trace 的推理提交区间起点排序。

![B1/S1 与 B1/S3 的 CUDA stream 时间线和跨请求 TensorRT kernel 重叠](results/nsight-slot-overlap.svg)

B1/S3 的前三个请求在饱和离线队列启动时被不同 worker 取走。以下时间相对该 trace 的首条 GPU kernel，起止点取每个请求的首条和末条 TensorRT kernel。

| 请求 | CUDA stream | 首条网络 kernel 开始（ms） | 末条网络 kernel 结束（ms） |
| --- | ---: | ---: | ---: |
| Req 1 | 15 | 0.06 | 53.22 |
| Req 2 | 49 | 3.11 | 54.84 |
| Req 3 | 83 | 4.08 | 55.73 |

三个请求的首末区间有 **49.15 ms** 的交集。按实际 kernel 区间累计，前三个请求中恰好两条 stream 同时执行网络 kernel 的时间为 **13.91 ms**，三条同时执行的时间为 **9.28 ms**。

下表统计完整采集。重叠比例的分母为至少一个相应类别 kernel 活跃的时间。无 kernel 活动的时间在首条到末条 kernel 之间统计。

| 32 张 trace 指标 | B1/S1 | B1/S3 |
| --- | ---: | ---: |
| CUDA stream 数（有 kernel） | 1 | 3 |
| 首条到末条 kernel 的时间窗口（ms） | 704.10 | 607.37 |
| 无 kernel 活动时间（ms） | 52.11 | 7.58 |
| 多 stream kernel 重叠（整条流水线） | 0.0% | 38.0% |
| 多 stream kernel 重叠（仅 TensorRT 网络） | 0.00% | 37.32% |
| 单请求 TensorRT GPU 投影跨度中位数（ms） | 19.95 | 53.22 |
| `cudaStreamSynchronize` 次数 | 32 | 32 |

两组各执行 16,608 条 kernel，其中 16,416 条关联到 TensorRT 推理。B1/S3 缩短了完成整组 GPU 工作的时间，每个请求的网络投影跨度则变长。GPU 投影跨度包含首末 GPU 操作之间的间隙。 [B1/S1 原始报告](results/trace/b1s1.nsys-rep) · [B1/S3 原始报告](results/trace/b1s3.nsys-rep) · [全流水线重叠汇总](results/trace/b1s3.overlap.json)

## RTX 4090 性能与准确率

### 500 张离线筛选

由 `ecf9e25` 运行，表中为三轮中位数，每轮 500 张、零失败。相同 JPEG 清单在计时前预载，每个 slot 预热 20 批。逐图 P95 从派发到 slot 计到 CPU 结果就绪。显存为 NVML 每 20 ms 采样到的推理进程峰值。

| 配置 | 吞吐（图/s） | 逐图 P95（ms） | 进程显存峰值（GiB） |
| --- | ---: | ---: | ---: |
| B1/S1 | 43.67 | 23.74 | 1.24 |
| B1/S2 | 50.89 | 41.93 | 1.79 |
| B1/S3 | 53.36 | 58.54 | 2.34 |
| B1/S4 | 53.08 | 77.74 | 2.89 |

同轮 Stage 4 原生 B1/S1 为 43.77 图/s，新调度器 B1/S1 为 43.67 图/s。以新调度器 B1/S1 为对照，B1/S3 吞吐提高 22.2%，S4 的延迟和显存进一步上升，因此选 S3 进入离线长测。 [逐轮结果（含同轮 batch 候选）](../stage6/results/screen/summary.json)

![500 张筛选的 B1/S1–S4 吞吐、逐图 P95 和进程显存](results/comparison.svg)

### 离线长测

在 `c8d4796` 上，同一份 500 张清单重复 35 次。每配置三轮，每轮 17,500 张，轮换运行顺序。稳态吞吐取总完成时间中间 90% 的窗口。表中为三轮中位数。

| 配置 | 稳态吞吐（图/s） | 总完成时间（s） | 逐图 P95（ms） | 进程显存峰值（GiB） | 三轮失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1/S1 | 43.74 | 400.18 | 23.68 | 1.24 | 0 / 52,500 |
| B1/S3 | 53.22 | 328.78 | 58.31 | 2.34 | 0 / 52,500 |

B1/S3 的稳态吞吐提高 **21.7%**，固定任务总完成时间缩短 **17.8%**。 [逐轮结果](results/offline/summary.json) · [逐请求记录校验](results/offline/raw.sha256)

### 正确性

32 张逐图对齐使用 Stage 4 的类别一致率、segment 配对和 IoU 阈值，B1/S1、S2、S3、S4 均通过。B1/S1、S2、S3 的正式评估在 `c8d4796` 上运行，各覆盖 5,000 个唯一 COCO 预测 ID。三组 PQ / PQ_th / PQ_st 均为 **53.186256 / 59.027261 / 44.369645**，与 Stage 4 一致。 [S1/S2/S4 对齐](../stage6/results/check/console.log) · [S3 对齐](../stage6/results/check-b8/b1s3/validation.json) · [全量 PQ](results/pq/)

## 复现入口

在 RTX 4090 上使用干净 checkout。沿用 [Stage 3 参考镜像](../stage3/results/environment.json)、COCO 数据、原始权重和固定 B1 engine。`previous` 指向 [Stage 4 正式运行目录](../stage4/README.md#复现)，需包含 `perf-native-r1` 和 `native-align-fixed`。每个运行 ID 只使用一次。以下命令使用两阶段共用的 [运行入口](run.sh)。

```bash
export MASKDINO_IMAGE=maskdino:stage1
data=/path/to/coco2017-panoptic-val
engine=/path/to/stage3/runs/build-fp32/network.engine
previous=/path/to/stage4/runs

bash stage5/run.sh "$data" "$previous" "$engine" slots-check check 1s1 1s2 1s3 1s4
bash stage5/run.sh "$data" "$previous" "$engine" slots-screen screen 1s1 1s2 1s3 1s4
bash stage5/run.sh "$data" "$previous" "$engine" slots-offline offline 1s1 1s3 --repeats 35
bash stage5/run.sh "$data" "$previous" "$engine" slots-pq pq 1s1 1s2 1s3
bash stage5/run.sh "$data" "$previous" "$engine" slots-trace trace 1s1 1s3
```

从已有报告重绘时间线只需离线导出 SQLite。以下命令使用 Nsight Systems CLI 和 Python 3。

```bash
for config in b1s1 b1s3; do
  nsys export --type=sqlite --output="/tmp/$config.sqlite" "stage5/results/trace/$config.nsys-rep"
done
python3 -m stage5.trace_plot /tmp/b1s1.sqlite /tmp/b1s3.sqlite stage5/results/nsight-slot-overlap.svg
```

500 张筛选、长测、PQ 和 trace 沿用拆分前的结果，正式提交分别见各节。历史实验入口、日志与原始布局见 [验收 tag `stage5-batch-slots-accepted`](https://github.com/Parker-Lyu/MaskDINO-TensorRT/tree/stage5-batch-slots-accepted/stage5)。
