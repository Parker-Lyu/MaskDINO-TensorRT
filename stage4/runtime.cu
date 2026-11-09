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

NetworkEngine::NetworkEngine(const std::string& engine_path) {
    std::ifstream input(engine_path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) throw std::runtime_error("Cannot open engine " + engine_path);
    std::vector<char> plan(size_t(input.tellg()));
    input.seekg(0);
    if (!input.read(plan.data(), plan.size())) throw std::runtime_error("Cannot read engine " + engine_path);
    static Logger logger;
    if (!initLibNvInferPlugins(&logger, "")) throw std::runtime_error("TensorRT plugin initialization failed");
    runtime_.reset(createInferRuntime(logger));
    if (!runtime_) throw std::runtime_error("TensorRT runtime creation failed");
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_ || engine_->getNbIOTensors() != 4 || engine_->getNbAuxStreams() != 0)
        throw std::runtime_error("Expected an FP32 engine with four tensors and no auxiliary streams");
}

Pipeline::Pipeline(const std::string& engine_path, const int* things)
    : Pipeline(std::make_shared<NetworkEngine>(engine_path), things, 1) {}

Pipeline::Pipeline(std::shared_ptr<NetworkEngine> engine, const int* things, int batch)
    : engine_(std::move(engine)), batch_(batch), preprocess_(new Preprocess[batch]),
      postprocess_(new Postprocess[batch]), geometries_(batch) {
    const auto& plan = engine_->engine_;
    context_.reset(plan->createExecutionContext());
    if (!context_) throw std::runtime_error("TensorRT context creation failed");
    auto bind = [&](const char* name, TensorIOMode mode, std::initializer_list<int> shape, Buffer<float>& buffer) {
        const auto dims = plan->getTensorShape(name);
        if (dims.nbDims != int(shape.size()) || !std::equal(shape.begin(), shape.end(), dims.d) ||
            plan->getTensorDataType(name) != DataType::kFLOAT ||
            plan->getTensorIOMode(name) != mode || plan->getTensorFormat(name) != TensorFormat::kLINEAR)
            throw std::runtime_error(std::string("Unexpected tensor layout: ") + name);
        size_t count = 1;
        for (int size : shape) count *= size;
        buffer.reserve(count);
        if (!context_->setTensorAddress(name, buffer.get()))
            throw std::runtime_error(std::string("Tensor binding failed: ") + name);
    };
    bind("image", TensorIOMode::kINPUT, {batch_, 3, side, side}, image_);
    bind("logits", TensorIOMode::kOUTPUT, {batch_, queries, classes}, logits_);
    bind("masks", TensorIOMode::kOUTPUT, {batch_, queries, mask_side, mask_side}, masks_);
    bind("boxes", TensorIOMode::kOUTPUT, {batch_, queries, 4}, boxes_);
    things_.reserve(classes);
    check(cudaMemcpy(things_.get(), things, classes * sizeof(int), cudaMemcpyHostToDevice));
}

Pipeline::~Pipeline() {
    cudaStreamSynchronize(stream_.value);
}

void Pipeline::run(const unsigned char* jpeg, size_t length, int height, int width, bool timing, bool stage_waits) {
    if (batch_ != 1) throw std::logic_error("Single-image call requires a B1 pipeline");
    const JpegView image{jpeg, length, height, width};
    run_batch(&image, timing, stage_waits);
}

void Pipeline::run_batch(const JpegView* images, bool timing, bool stage_waits) {
    const auto stream = stream_.value;
    for (int i = 0; i < batch_; ++i) {
        const auto& item = images[i];
        auto& geometry = geometries_[i];
        geometry = preprocess_[i].run(item.data, item.length, image_.get() + size_t(i) * 3 * side * side, stream);
        if (geometry.height != item.height || geometry.width != item.width)
            throw std::runtime_error("Decoded image dimensions disagree with the manifest");
    }
    if (stage_waits) {
        Range range("CPU / wait for preprocess");
        check(cudaStreamSynchronize(stream));
    }
    for (int i = 0; i < batch_; ++i)
        postprocess_[i].reserve(size_t(images[i].height) * images[i].width);
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
        for (int i = 0; i < batch_; ++i)
            postprocess_[i].run(logits_.get() + size_t(i) * queries * classes,
                                masks_.get() + size_t(i) * queries * mask_side * mask_side,
                                things_.get(), geometries_[i], stream);
    }
    if (stage_waits) {
        Range range("CPU / wait for postprocess");
        check(cudaStreamSynchronize(stream));
    }
    for (int i = 0; i < batch_; ++i)
        postprocess_[i].download(geometries_[i], stream, stage_waits, i == batch_ - 1);
}

float Pipeline::network_ms() {
    float ms = 0;
    check(cudaEventElapsedTime(&ms, begin_.value, end_.value));
    return ms;
}
