#include "runtime.cuh"

#include <NvInferPlugin.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace nvinfer1;

void Logger::log(Severity severity, const char* message) noexcept {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "TensorRT: %s\n", message);
}

Pipeline::Pipeline(const std::string& engine_path, const int* things) {
    std::ifstream input(engine_path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) throw std::runtime_error("Cannot open engine " + engine_path);
    std::vector<char> plan(size_t(input.tellg()));
    input.seekg(0);
    if (!input.read(plan.data(), plan.size())) throw std::runtime_error("Cannot read engine " + engine_path);
    if (!initLibNvInferPlugins(&logger_, "")) throw std::runtime_error("TensorRT plugin initialization failed");
    runtime_.reset(createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("TensorRT runtime creation failed");
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_ || engine_->getNbIOTensors() != 4 || engine_->getNbAuxStreams() != 0)
        throw std::runtime_error("Expected the Stage 3 FP32 engine with four tensors and no auxiliary streams");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("TensorRT context creation failed");
    auto bind = [&](const char* name, TensorIOMode mode, std::initializer_list<int> shape, Buffer<float>& buffer) {
        const auto dims = engine_->getTensorShape(name);
        if (dims.nbDims != int(shape.size()) || !std::equal(shape.begin(), shape.end(), dims.d) ||
            engine_->getTensorDataType(name) != DataType::kFLOAT ||
            engine_->getTensorIOMode(name) != mode || engine_->getTensorFormat(name) != TensorFormat::kLINEAR)
            throw std::runtime_error(std::string("Unexpected tensor layout: ") + name);
        size_t count = 1;
        for (int size : shape) count *= size;
        buffer.reserve(count);
        if (!context_->setTensorAddress(name, buffer.get()))
            throw std::runtime_error(std::string("Tensor binding failed: ") + name);
    };
    bind("image", TensorIOMode::kINPUT, {1, 3, side, side}, image_);
    bind("logits", TensorIOMode::kOUTPUT, {1, queries, classes}, logits_);
    bind("masks", TensorIOMode::kOUTPUT, {1, queries, mask_side, mask_side}, masks_);
    bind("boxes", TensorIOMode::kOUTPUT, {1, queries, 4}, boxes_);
    things_.reserve(classes);
    check(cudaMemcpy(things_.get(), things, classes * sizeof(int), cudaMemcpyHostToDevice));
}

void Pipeline::run(const unsigned char* jpeg, size_t length, int height, int width, bool timing, bool stage_waits) {
    const auto stream = stream_.value;
    const auto geometry = preprocess_.run(jpeg, length, image_.get(), stream);
    if (geometry.height != height || geometry.width != width)
        throw std::runtime_error("Decoded image dimensions disagree with the manifest");
    if (stage_waits) {
        Range range("CPU / wait for preprocess");
        check(cudaStreamSynchronize(stream));
    }
    postprocess_.reserve(size_t(height) * width);
    if (timing) check(cudaEventRecord(begin_.value, stream));
    {
        Range range("TensorRT inference");
        if (!context_->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueue failed");
    }
    if (timing) check(cudaEventRecord(end_.value, stream));
    if (stage_waits) {
        Range range("CPU / wait for network");
        check(cudaStreamSynchronize(stream));
    }
    {
        Range range("postprocess");
        postprocess_.run(logits_.get(), masks_.get(), things_.get(), geometry, stream);
    }
    if (stage_waits) {
        Range range("CPU / wait for postprocess");
        check(cudaStreamSynchronize(stream));
    }
    postprocess_.download(geometry, stream, stage_waits);
}

float Pipeline::network_ms() {
    float ms = 0;
    check(cudaEventElapsedTime(&ms, begin_.value, end_.value));
    return ms;
}
