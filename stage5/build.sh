#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:?Specify an output directory}
mkdir -p "$output"
nvcc -std=c++17 -O3 -arch=sm_89 --fmad=false -rdc=true -dlto -Xcompiler=-ffp-contract=off,-pthread \
    "$root"/stage4/{preprocess,restore,postprocess,runtime}.cu "$root/stage5/main.cu" \
    -lnvinfer -lnvinfer_plugin -ljpeg -L"${CUDA_HOME:-/usr/local/cuda}/lib64/stubs" -lnvidia-ml \
    -o "$output/maskdino-slots"
