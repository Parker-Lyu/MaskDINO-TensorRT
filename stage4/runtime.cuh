#pragma once

#include "postprocess.cuh"

#include <NvInfer.h>
#include <memory>
#include <string>

struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override;
};

class Pipeline {
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Stream stream_;
    Event begin_, end_;
    Buffer<float> image_, logits_, masks_, boxes_;
    Buffer<int> things_;
    Preprocess preprocess_;
    Postprocess postprocess_;
public:
    Pipeline(const std::string& engine_path, const int* things);
    void run(const unsigned char* jpeg, size_t length, int height, int width, bool timing, bool stage_waits);
    float network_ms();
    const int* map() { return postprocess_.map(); }
    const Segments& segments() { return postprocess_.segments(); }
};
