#pragma once

#include "preprocess.cuh"

constexpr int queries = 300;
constexpr int classes = 133;
constexpr int mask_side = 256;

struct Segment {
    int id;
    int category;
    int isthing;
};

struct Segments {
    int count;
    Segment entries[queries];
};

struct Candidates {
    int count;
    int query[queries];
    int label[queries];
    float score[queries];
};

struct Areas {
    int original[queries];
    int assigned[queries];
    int intersection[queries];
};

__device__ float restored_mask(const float* mask, Geometry geometry, int x, int y);

class Postprocess {
    Buffer<Candidates> candidates_;
    Buffer<Areas> areas_;
    Buffer<int> winner_, ids_, map_;
    Buffer<Segments> segments_;
    Buffer<int, true> host_map_;
    Buffer<Segments, true> host_segments_;
public:
    void reserve(size_t pixels);
    void run(const float* logits, const float* masks, const int* things, Geometry geometry, cudaStream_t stream);
    void download(Geometry geometry, cudaStream_t stream, bool stage_waits = false, bool wait = true);
    const int* map() { return host_map_.get(); }
    const Segments& segments() { return *host_segments_.get(); }
};
