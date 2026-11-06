#pragma once

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

constexpr int side = 1024;

inline void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct Range {
    explicit Range(const char* name) { nvtxRangePushA(name); }
    ~Range() { nvtxRangePop(); }
};

template<class T, bool pinned = false>
class Buffer {
    T* data_ = nullptr;
    size_t capacity_ = 0;
public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { release(); }
    void reserve(size_t count) {
        if (count <= capacity_) return;
        T* next = nullptr;
        if constexpr (pinned) check(cudaMallocHost(&next, count * sizeof(T)));
        else check(cudaMalloc(&next, count * sizeof(T)));
        release();
        data_ = next;
        capacity_ = count;
    }
    T* get() { return data_; }
private:
    void release() {
        if (!data_) return;
        if constexpr (pinned) cudaFreeHost(data_);
        else cudaFree(data_);
    }
};

struct Geometry {
    int height;
    int width;
    int valid_height;
    int valid_width;
};

struct AxisTap {
    int first;
    int count;
    int offset;
};

class Preprocess {
    Buffer<unsigned char, true> rgb_;
    Buffer<unsigned char> device_rgb_, intermediate_;
    Buffer<AxisTap, true> taps_;
    Buffer<int, true> coefficients_;
    Buffer<AxisTap> device_taps_;
    Buffer<int> device_coefficients_;
public:
    Geometry run(const unsigned char* jpeg, size_t length, float* image, cudaStream_t stream);
    unsigned char* decoded() { return rgb_.get(); }
};
