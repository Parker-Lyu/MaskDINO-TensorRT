#include "runtime.cuh"

#include <cuda_profiler_api.h>
#include <nvml.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

struct Request {
    int id, height, width;
    std::string path;
    std::vector<unsigned char> jpeg;
};

std::vector<unsigned char> load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) throw std::runtime_error("Cannot open " + path);
    std::vector<unsigned char> bytes(size_t(file.tellg()));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("Cannot read " + path);
    return bytes;
}

void nvml_check(nvmlReturn_t result) {
    if (result != NVML_SUCCESS) throw std::runtime_error(nvmlErrorString(result));
}

class ProcessMemory {
    nvmlDevice_t device_{};
    std::atomic<bool> stop_{false};
    std::thread worker_;
    unsigned long long peak_ = 0;
    bool failed_ = false;
    unsigned long long used() {
        std::vector<nvmlProcessInfo_t> processes(32);
        unsigned int count = processes.size();
        auto result = nvmlDeviceGetComputeRunningProcesses(device_, &count, processes.data());
        if (result == NVML_ERROR_INSUFFICIENT_SIZE) {
            processes.resize(count);
            result = nvmlDeviceGetComputeRunningProcesses(device_, &count, processes.data());
        }
        nvml_check(result);
        for (unsigned int i = 0; i < count; ++i) {
            const auto& process = processes[i];
            if (process.pid == static_cast<unsigned int>(getpid()) &&
                process.usedGpuMemory != static_cast<unsigned long long>(NVML_VALUE_NOT_AVAILABLE)) return process.usedGpuMemory;
        }
        throw std::runtime_error("NVML could not find the inference process. Use Docker --pid=host.");
    }
public:
    ProcessMemory() {
        nvml_check(nvmlInit());
        try {
            char pci[32];
            check(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0));
            nvml_check(nvmlDeviceGetHandleByPciBusId(pci, &device_));
            peak_ = used();
            worker_ = std::thread([this] {
                try {
                    while (!stop_.load()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        peak_ = std::max(peak_, used());
                    }
                } catch (...) {
                    failed_ = true;
                }
            });
        } catch (...) {
            nvmlShutdown();
            throw;
        }
    }
    ~ProcessMemory() {
        stop_.store(true);
        if (worker_.joinable()) worker_.join();
        nvmlShutdown();
    }
    unsigned long long finish() {
        stop_.store(true);
        worker_.join();
        if (failed_) throw std::runtime_error("NVML process sampling failed");
        return std::max(peak_, used());
    }
};

void save_prediction(const fs::path& output, const Request& request, Pipeline& pipeline) {
    auto stem = output / std::to_string(request.id);
    std::ofstream map(stem.string() + ".bin", std::ios::binary);
    map.write(reinterpret_cast<const char*>(pipeline.map()),
              size_t(request.height) * request.width * sizeof(int));
    map.close();
    std::ofstream metadata(stem.string() + ".tsv");
    const auto& segments = pipeline.segments();
    for (int i = 0; i < segments.count; ++i) {
        const auto& entry = segments.entries[i];
        metadata << entry.id << '\t' << entry.category << '\t' << entry.isthing << '\n';
    }
    metadata.close();
    if (!map || !metadata) throw std::runtime_error("Could not write prediction " + stem.string());
}

void run(int argc, char** argv) {
    if (argc != 6)
        throw std::runtime_error("Usage: maskdino-native ENGINE THINGS MANIFEST predict|benchmark|trace|trace-stage-waits OUTPUT");
    const std::string mode = argv[4];
    const bool stage_waits = mode == "trace-stage-waits";
    const bool benchmark = mode == "benchmark", trace = mode == "trace" || stage_waits;
    if (!benchmark && !trace && mode != "predict") throw std::runtime_error("Unknown mode " + mode);
    const fs::path output = argv[5];
    std::array<int, classes> things{};
    std::ifstream categories(argv[2]);
    int category;
    while (categories >> category) {
        if (category < 0 || category >= classes) throw std::runtime_error("Invalid thing category");
        things[category] = 1;
    }
    if (!categories.eof() || std::count(things.begin(), things.end(), 1) != 80)
        throw std::runtime_error("Expected 80 distinct COCO thing categories");
    std::ifstream manifest(argv[3]);
    std::vector<Request> requests;
    Request request{};
    while (manifest >> request.id >> request.height >> request.width >> std::quoted(request.path)) {
        if (request.height <= 0 || request.width <= 0 || request.height > 16384 || request.width > 16384)
            throw std::runtime_error("Invalid manifest dimensions");
        if (benchmark || trace) request.jpeg = load(request.path);
        requests.push_back(std::move(request));
    }
    if (!manifest.eof() || requests.empty()) throw std::runtime_error("Invalid or empty manifest");

    Pipeline pipeline(argv[1], things.data());
    if (benchmark || trace) {
        const auto& first = requests.front();
        for (int i = 0; i < 20; ++i)
            pipeline.run(first.jpeg.data(), first.jpeg.size(), first.height, first.width, false, stage_waits);
    }
    auto memory = benchmark ? std::make_unique<ProcessMemory>() : nullptr;
    struct Sample { int id; float network_ms; double e2e_ms; };
    std::vector<Sample> samples;
    samples.reserve(requests.size());
    if (mode == "predict") fs::create_directories(output / "predictions");
    if (trace) check(cudaProfilerStart());
    const auto start = Clock::now();
    for (auto& item : requests) {
        if (mode == "predict") item.jpeg = load(item.path);
        const auto name = "image " + std::to_string(item.id);
        const auto before = Clock::now();
        {
            Range range(name.c_str());
            pipeline.run(item.jpeg.data(), item.jpeg.size(), item.height, item.width, benchmark, stage_waits);
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - before).count();
        if (benchmark) samples.push_back({item.id, pipeline.network_ms(), ms});
        if (mode == "predict") {
            save_prediction(output / "predictions", item, pipeline);
            item.jpeg.clear();
            item.jpeg.shrink_to_fit();
        }
    }
    const double loop_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (trace) check(cudaProfilerStop());
    const auto peak = memory ? memory->finish() : 0;
    std::ofstream metrics(output / "native.tsv");
    metrics << std::setprecision(12) << requests.size() << '\t' << loop_seconds << '\t' << peak << '\n';
    for (const auto& sample : samples)
        metrics << sample.id << '\t' << sample.network_ms << '\t' << sample.e2e_ms << '\n';
    metrics.close();
    if (!metrics) throw std::runtime_error("Could not write native timing record");
    std::cout << "Completed " << requests.size() << " images in " << loop_seconds << " s\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        run(argc, argv);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
