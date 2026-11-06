#pragma once

#include "runtime.cuh"

#include <nvml.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <vector>

namespace native_bench {
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

struct Request {
    int id, height, width;
    std::string path;
    std::vector<unsigned char> jpeg;
};

inline std::vector<unsigned char> load(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0) throw std::runtime_error("Cannot open " + path);
    std::vector<unsigned char> bytes(size_t(file.tellg()));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("Cannot read " + path);
    return bytes;
}

inline void nvml_check(nvmlReturn_t result) {
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

inline std::array<int, classes> read_things(const char* path) {
    std::array<int, classes> things{};
    std::ifstream categories(path);
    int category;
    while (categories >> category) {
        if (category < 0 || category >= classes) throw std::runtime_error("Invalid thing category");
        things[category] = 1;
    }
    if (!categories.eof() || std::count(things.begin(), things.end(), 1) != 80)
        throw std::runtime_error("Expected 80 distinct COCO thing categories");
    return things;
}

inline std::vector<Request> read_requests(const char* path, bool preload) {
    std::ifstream manifest(path);
    std::vector<Request> requests;
    Request request{};
    while (manifest >> request.id >> request.height >> request.width >> std::quoted(request.path)) {
        if (request.height <= 0 || request.width <= 0 || request.height > 16384 || request.width > 16384)
            throw std::runtime_error("Invalid manifest dimensions");
        if (preload) request.jpeg = load(request.path);
        requests.push_back(std::move(request));
    }
    if (!manifest.eof() || requests.empty()) throw std::runtime_error("Invalid or empty manifest");

    return requests;
}

}  // namespace native_bench
