#pragma once

#include "postprocess.cuh"

#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>

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

class NetworkEngine {
    friend class Pipeline;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
public:
    explicit NetworkEngine(const std::string& path);
};

struct JpegView {
    const unsigned char* data;
    size_t length;
    int height, width;
};

class Pipeline {
    std::shared_ptr<NetworkEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Stream stream_;
    Event begin_, end_;
    Buffer<float> image_, logits_, masks_, boxes_;
    Buffer<int> things_;
    int batch_;
    std::unique_ptr<Preprocess[]> preprocess_;
    std::unique_ptr<Postprocess[]> postprocess_;
    std::vector<Geometry> geometries_;
public:
    Pipeline(const std::string& engine_path, const int* things);
    Pipeline(std::shared_ptr<NetworkEngine> engine, const int* things, int batch);
    ~Pipeline();
    void run_batch(const JpegView* images, bool timing, bool stage_waits = false);
    void run(const unsigned char* jpeg, size_t length, int height, int width, bool timing, bool stage_waits);
    float network_ms();
    const int* map(int index = 0) { return postprocess_[index].map(); }
    const Segments& segments(int index = 0) { return postprocess_[index].segments(); }
};
