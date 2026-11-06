// Test adapter. The production runtime calls Preprocess directly.
#include "preprocess.cuh"

#include <cstdio>
#include <cstring>

extern "C" int stage4_preprocess(const unsigned char* jpeg, size_t length,
                                  Geometry* geometry, unsigned char* rgb, float* image) {
    try {
        Preprocess preprocess;
        Buffer<float> device_image;
        device_image.reserve(3 * side * side);
        *geometry = preprocess.run(jpeg, length, device_image.get(), nullptr);
        check(cudaMemcpy(image, device_image.get(), 3 * side * side * sizeof(float), cudaMemcpyDeviceToHost));
        std::memcpy(rgb, preprocess.decoded(), size_t(geometry->height) * geometry->width * 3);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
