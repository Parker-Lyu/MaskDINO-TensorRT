#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
output=${1:?Specify an output directory}
mkdir -p "$output"
nvcc -std=c++17 -O3 -arch=sm_89 --fmad=false -Xcompiler=-fPIC,-ffp-contract=off \
    --shared preprocess.cu preprocess_check.cu -ljpeg -o "$output/preprocess-check.so"
nvcc -std=c++17 -O3 -arch=sm_89 --fmad=false -rdc=true -Xcompiler=-fPIC \
    --shared restore.cu restore_check.cu -o "$output/restore-check.so"
nvcc -std=c++17 -O3 -arch=sm_89 --fmad=false -rdc=true -dlto -Xcompiler=-ffp-contract=off \
    preprocess.cu restore.cu postprocess.cu runtime.cu main.cu \
    -lnvinfer -lnvinfer_plugin -ljpeg -L"${CUDA_HOME:-/usr/local/cuda}/lib64/stubs" -lnvidia-ml \
    -o "$output/maskdino-native"
