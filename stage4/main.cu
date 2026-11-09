#include "native_bench.cuh"

#include <cuda_profiler_api.h>
#include <iostream>

namespace {
using namespace native_bench;

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
    const auto things = read_things(argv[2]);
    auto requests = read_requests(argv[3], benchmark || trace);
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
