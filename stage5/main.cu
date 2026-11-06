#include "../stage4/native_bench.cuh"

#include <cuda_profiler_api.h>

#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>

namespace {
using namespace native_bench;
using Milliseconds = std::chrono::duration<double, std::milli>;

struct Sample {
    Clock::time_point arrival, queued, dispatched, ready;
    size_t dispatch_depth = 0, arrival_depth = 0, group = 0;
    int slot = -1, batch = 0;
    float network_ms = 0;
    double batch_wait_ms = 0;
    bool rejected = false;
};

struct Prediction {
    std::vector<int> map;
    Segments segments;
};

void save(const fs::path& output, int id, const Prediction& prediction) {
    const auto stem = (output / std::to_string(id)).string();
    std::ofstream map(stem + ".bin", std::ios::binary);
    map.write(reinterpret_cast<const char*>(prediction.map.data()), prediction.map.size() * sizeof(int));
    map.close();
    std::ofstream metadata(stem + ".tsv");
    for (int i = 0; i < prediction.segments.count; ++i) {
        const auto& segment = prediction.segments.entries[i];
        metadata << segment.id << '\t' << segment.category << '\t' << segment.isthing << '\n';
    }
    metadata.close();
    if (!map || !metadata) throw std::runtime_error("Failed to save prediction " + stem);
}

std::vector<double> read_arrivals(const char* path) {
    std::ifstream file(path);
    std::vector<double> arrivals;
    double ms;
    while (file >> ms) {
        if (!std::isfinite(ms) || ms < 0 || ms > 86400000 || (!arrivals.empty() && ms < arrivals.back()))
            throw std::runtime_error("Invalid arrival time");
        arrivals.push_back(ms);
    }
    if (!file.eof() || arrivals.empty()) throw std::runtime_error("Invalid arrival file");
    return arrivals;
}

JpegView jpeg_view(const Request& image) {
    return {image.jpeg.data(), image.jpeg.size(), image.height, image.width};
}

int integer(const char* text) {
    size_t consumed;
    const int value = std::stoi(text, &consumed);
    if (text[consumed] || value <= 0) throw std::runtime_error("Expected a positive integer");
    return value;
}

struct Slot {
    Pipeline full;
    std::unique_ptr<Pipeline> single;
    Slot(const std::shared_ptr<NetworkEngine>& engine, const std::shared_ptr<NetworkEngine>& fallback,
         const int* things, int batch)
        : full(engine, things, batch) {
        if (fallback) single = std::make_unique<Pipeline>(fallback, things, 1);
    }
};

class Scheduler {
    const std::vector<Request>& images_;
    const int batch_;
    const Milliseconds delay_;
    const size_t capacity_;
    const bool predict_;
    const fs::path predictions_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<size_t> queue_;
    Clock::time_point batch_wait_started_{};
    size_t next_ = 0, group_ = 0;
    bool closed_ = false;
    std::exception_ptr error_;

public:
    std::vector<Sample> samples;
    Scheduler(const std::vector<Request>& images, size_t count, int batch, double delay, size_t capacity,
              bool predict, const fs::path& output)
        : images_(images), batch_(batch), delay_(delay), capacity_(capacity),
          predict_(predict), predictions_(output / "predictions"), samples(count) {
        if (predict_) fs::create_directories(predictions_);
    }

    void arrive(size_t index, Clock::time_point planned) {
        std::lock_guard lock(mutex_);
        auto& sample = samples[index];
        sample.arrival = planned;
        sample.queued = Clock::now();
        sample.arrival_depth = queue_.size();
        if (queue_.size() == capacity_) {
            sample.rejected = true;
        } else {
            queue_.push_back(index);
            changed_.notify_all();
        }
    }

    void close() {
        std::lock_guard lock(mutex_);
        closed_ = true;
        changed_.notify_all();
    }

    void run(Slot& slot, int slot_id, bool online) {
        try {
            worker(slot, slot_id, online);
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!error_) error_ = std::current_exception();
            changed_.notify_all();
        }
    }

    void check_error() {
        std::lock_guard lock(mutex_);
        if (error_) std::rethrow_exception(error_);
    }

private:
    size_t dispatch(std::vector<size_t>& ids, int slot_id, bool online) {
        std::unique_lock lock(mutex_);
        if (online) {
            for (;;) {
                changed_.wait(lock, [&] { return error_ || closed_ || !queue_.empty(); });
                if (error_ || queue_.empty()) return 0;
                const auto deadline = samples[queue_.front()].arrival + delay_;
                if (closed_ || queue_.size() >= size_t(batch_) || Clock::now() >= deadline) break;
                if (batch_wait_started_ == Clock::time_point{}) batch_wait_started_ = Clock::now();
                changed_.wait_until(lock, deadline);
            }
        } else if (error_ || next_ == samples.size()) {
            return 0;
        }
        const auto depth = online ? queue_.size() : samples.size() - next_;
        const int count = depth >= size_t(batch_) ? batch_ : 1;
        const auto start = Clock::now();
        const size_t group = group_++;
        for (int i = 0; i < count; ++i) {
            if (online) {
                ids[i] = queue_.front();
                queue_.pop_front();
            } else {
                ids[i] = next_++;
                samples[ids[i]].arrival = samples[ids[i]].queued = start;
            }
            auto& sample = samples[ids[i]];
            sample.dispatched = start;
            if (batch_wait_started_ != Clock::time_point{})
                sample.batch_wait_ms = Milliseconds(start - std::max(sample.queued, batch_wait_started_)).count();
            sample.dispatch_depth = depth;
            sample.slot = slot_id;
            sample.batch = count;
            sample.group = group;
        }
        batch_wait_started_ = {};
        return count;
    }

    void worker(Slot& slot, int slot_id, bool online) {
        check(cudaSetDevice(0));
        std::vector<size_t> ids(batch_);
        std::vector<JpegView> views(batch_);
        std::vector<Prediction> results(batch_);
        while (const size_t count = dispatch(ids, slot_id, online)) {
            auto& pipeline = count == size_t(batch_) ? slot.full : *slot.single;
            for (size_t i = 0; i < count; ++i) views[i] = jpeg_view(images_[ids[i] % images_.size()]);
            pipeline.run_batch(views.data(), true);
            const float network_ms = pipeline.network_ms();
            for (size_t i = 0; i < count; ++i) {
                auto& result = results[i];
                const auto* map = pipeline.map(i);
                result.map.assign(map, map + size_t(views[i].height) * views[i].width);
                result.segments = pipeline.segments(i);
                auto& sample = samples[ids[i]];
                sample.ready = Clock::now();
                sample.network_ms = network_ms;
                if (predict_) save(predictions_, images_[ids[i]].id, result);
            }
        }
    }
};

void run(int argc, char** argv) {
    if (argc != 12)
        throw std::runtime_error("Usage: maskdino-slots ENGINE FALLBACK|- THINGS MANIFEST OUTPUT "
                                 "BATCH SLOTS benchmark|predict|trace|online|online-predict REPEATS|ARRIVALS DELAY_MS QUEUE_CAPACITY");
    const int batch = integer(argv[6]), slots = integer(argv[7]);
    if (batch > 8 || slots > 16) throw std::runtime_error("Batch or slot count exceeds supported range");
    const std::string mode = argv[8];
    const bool online = mode == "online" || mode == "online-predict";
    const bool predict = mode == "predict" || mode == "online-predict", trace = mode == "trace";
    if (!(online || predict || trace || mode == "benchmark")) throw std::runtime_error("Unknown mode");
    const size_t capacity = integer(argv[11]);
    size_t consumed;
    const double delay = std::stod(argv[10], &consumed);
    if (argv[10][consumed] || !std::isfinite(delay) || delay < 0 || delay > 60000)
        throw std::runtime_error("Invalid batch delay");
    const auto arrivals = online ? read_arrivals(argv[9]) : std::vector<double>{};
    const int repeats = online ? 1 : integer(argv[9]);
    const auto things = read_things(argv[3]);
    const auto images = read_requests(argv[4], true);
    const size_t count = online ? arrivals.size() : images.size() * repeats;
    if (predict) {
        std::set<int> ids;
        for (const auto& image : images) ids.insert(image.id);
        if (count != images.size() || ids.size() != images.size()) throw std::runtime_error("Prediction IDs must be unique");
    }
    const fs::path output = argv[5];
    fs::create_directories(output);

    auto engine = std::make_shared<NetworkEngine>(argv[1]);
    std::shared_ptr<NetworkEngine> fallback;
    if (batch > 1 && (online || count % batch))
        fallback = std::make_shared<NetworkEngine>(argv[2]);
    std::vector<std::unique_ptr<Slot>> workers;
    for (int i = 0; i < slots; ++i)
        workers.push_back(std::make_unique<Slot>(engine, fallback, things.data(), batch));

    std::vector<JpegView> warmup(batch);
    for (int i = 0; i < batch; ++i) warmup[i] = jpeg_view(images[i % images.size()]);
    if (!predict || online) {
        for (auto& worker : workers) {
            for (int i = 0; i < 20; ++i) {
                worker->full.run_batch(warmup.data(), false);
                if (worker->single) worker->single->run_batch(warmup.data(), false);
            }
        }
    }
    auto memory = predict || trace ? nullptr : std::make_unique<ProcessMemory>();
    Scheduler scheduler(images, count, batch, delay, capacity, predict, output);
    std::vector<std::thread> threads;
    threads.reserve(slots);
    if (trace) check(cudaProfilerStart());
    const auto start = Clock::now();
    try {
        for (int i = 0; i < slots; ++i)
            threads.emplace_back([&, i] { scheduler.run(*workers[i], i, online); });
        for (size_t i = 0; i < arrivals.size(); ++i) {
            const auto planned = start + std::chrono::duration_cast<Clock::duration>(Milliseconds(arrivals[i]));
            std::this_thread::sleep_until(planned);
            scheduler.check_error();
            scheduler.arrive(i, planned);
        }
    } catch (...) {
        scheduler.close();
        for (auto& thread : threads) thread.join();
        throw;
    }
    scheduler.close();
    for (auto& thread : threads) thread.join();
    scheduler.check_error();
    if (trace) check(cudaProfilerStop());
    const auto peak = memory ? memory->finish() : 0;
    size_t complete = 0;
    auto last = start;
    for (const auto& sample : scheduler.samples) {
        if (!sample.rejected) {
            ++complete;
            last = std::max(last, sample.ready);
        }
    }
    const double seconds = std::chrono::duration<double>(last - start).count();
    std::ofstream metrics(output / "requests.tsv");
    metrics << std::setprecision(12) << "request\timage\tarrival_ms\tqueued_ms\tdispatch_ms\tready_ms"
            "\tarrival_depth\tdispatch_depth\tslot\tbatch\tgroup\tnetwork_ms\tbatch_wait_ms\tstatus\n";
    auto relative = [&](auto timestamp) { return Milliseconds(timestamp - start).count(); };
    for (size_t i = 0; i < count; ++i) {
        const auto& s = scheduler.samples[i];
        metrics << i << '\t' << images[i % images.size()].id << '\t' << relative(s.arrival) << '\t'
                << relative(s.queued) << '\t';
        if (s.rejected) metrics << "\t\t" << s.arrival_depth << "\t\t\t\t\t\t\trejected\n";
        else metrics << relative(s.dispatched) << '\t' << relative(s.ready) << '\t'
                     << s.arrival_depth << '\t' << s.dispatch_depth << '\t' << s.slot << '\t'
                     << s.batch << '\t' << s.group << '\t' << s.network_ms << '\t' << s.batch_wait_ms << "\tok\n";
    }
    metrics.close();
    std::ofstream summary(output / "run.tsv");
    summary << std::setprecision(12) << complete << '\t' << count - complete << '\t' << seconds << '\t' << peak << '\n';
    summary.close();
    if (!metrics || !summary) throw std::runtime_error("Failed to write timing records");
    std::cout << "Completed " << complete << '/' << count << " images in " << seconds << " s\n";
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
