#include "postprocess.cuh"

__global__ void restore(const float* masks, float* output, Geometry geometry, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = geometry.width * geometry.height;
    if (i >= count * pixels) return;
    const int p = i % pixels;
    output[i] = restored_mask(masks + (i / pixels) * mask_side * mask_side, geometry,
                              p % geometry.width, p / geometry.width);
}

extern "C" int stage4_restore(const float* masks, float* output, Geometry geometry,
                              int count, cudaStream_t stream) {
    const int length = count * geometry.width * geometry.height;
    restore<<<(length + 255) / 256, 256, 0, stream>>>(masks, output, geometry, count);
    return cudaGetLastError();
}
