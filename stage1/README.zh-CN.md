# Stage 1 · 复现原始模型精度

[English](README.md) · **简体中文**

使用官方 MaskDINO 提交 `3831d8514a3728535ace8d4ecc7d28044c42dd14`、R50 50 epoch panoptic 配置和对应权重。输入短边缩放到 800、长边上限 1333，保留官方配置的 panoptic、instance 和 semantic 三个输出分支。

## 复现

在 Linux x86_64 GPU 主机上准备 Docker、NVIDIA Container Toolkit 和兼容 CUDA 13 的驱动。当前构建针对 RTX 4090 的 SM 8.9。按[数据与权重说明](../docs/experiment-environment.zh-CN.md)准备 COCO 文件。下载脚本会获取权重并校验哈希。

从仓库根目录执行：

```bash
python3 maskdino/scripts/fetch.py assets
docker build --network=host -f maskdino/Dockerfile -t maskdino:stage1 .
bash stage1/run.sh \
  /path/to/coco2017-panoptic-val \
  /path/to/instances_val2017.json
```

[Dockerfile](../maskdino/Dockerfile) 指定基础镜像标签并固定依赖版本，并应用 [CUDA 扩展兼容补丁](../maskdino/patches/maskdino-pytorch.patch)。基础镜像的参考 ID 见 [依赖锁文件](../maskdino/dependencies.lock.json)。脚本默认使用 `maskdino:stage1`，也可通过 `MASKDINO_IMAGE` 指定使用同一构建配方的其他镜像。

脚本校验权重和 panoptic 标注 JSON 的 SHA-256，将输入只读挂载，调用官方预处理脚本生成语义标签，并运行官方 evaluator。日志和预测写到 `stage1/runs/<run-id>/`。

## 结果

2026-11-06 从项目提交 `6eb01c17745e10756558464f3f85b35e76bbca09` 在 RTX 4090 上重跑官方评估，预测覆盖 COCO 2017 val 全部 5,000 张图。PQ 为 **52.9567**，PQ Things 为 **58.9645**，PQ Stuff 为 **43.8884**，与原始评估一致。

[重跑日志摘录和覆盖校验](results/official-r50-20261106-6eb01c1.log) · [原始评估日志摘录](results/official-r50-20261106.log)
