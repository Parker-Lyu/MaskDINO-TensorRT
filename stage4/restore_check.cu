#include "postprocess.cuh"

#include <algorithm>
#include <cstdio>
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

// Diagnostic adapter using exactly the same device tensors as the reference.
extern "C" int stage4_panoptic(const float* logits, const float* masks, const int* things,
                               Geometry geometry, cudaStream_t stream, int* map, Segments* segments) {
    try {
        Postprocess postprocess;
        postprocess.run(logits, masks, things, geometry, stream);
        postprocess.download(geometry, stream);
        std::copy_n(postprocess.map(), size_t(geometry.height) * geometry.width, map);
        *segments = postprocess.segments();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
