#include "postprocess.cuh"

namespace {
constexpr int threads = 256;

__device__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__global__ void select(const float* logits, Candidates* candidates) {
    __shared__ int labels[queries];
    __shared__ float scores[queries];
    const int query = threadIdx.x;
    const auto* row = logits + query * classes;
    float best = -1;
    int label = 0;
    for (int c = 0; c < classes; ++c) {
        const float score = sigmoid(row[c]);
        if (score > best) {
            best = score;
            label = c;
        }
    }
    labels[query] = label;
    scores[query] = 0;
    if (label != classes - 1 && best > 0.25f) {
        float sum = 0;
        const float center = best / 0.06f;
        for (int c = 0; c < classes; ++c)
            sum += expf(sigmoid(row[c]) / 0.06f - center);
        scores[query] = 1.0f / sum;
    }
    __syncthreads();
    if (query != 0) return;
    int count = 0;
    for (int q = 0; q < queries; ++q) {
        if (!scores[q]) continue;
        candidates->query[count] = q;
        candidates->label[count] = labels[q];
        candidates->score[count++] = scores[q];
    }
    candidates->count = count;
}

__global__ void assign(const float* masks, const Candidates* candidates, Geometry geometry,
                       Areas* areas, int* winner) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = p < geometry.height * geometry.width;
    const unsigned lanes = __ballot_sync(0xffffffff, active);
    if (!active) return;
    const int lane = threadIdx.x & 31;
    const int x = p % geometry.width, y = p / geometry.width;
    float best = -1;
    int selected = -1;
    bool inside = false;
    for (int k = 0; k < candidates->count; ++k) {
        const auto* mask = masks + candidates->query[k] * mask_side * mask_side;
        const float probability = sigmoid(restored_mask(mask, geometry, x, y));
        const bool above = probability >= 0.5f;
        const unsigned positives = __ballot_sync(lanes, above);
        if (lane == 0 && positives) atomicAdd(&areas->original[k], __popc(positives));
        const float score = candidates->score[k] * probability;
        if (score > best) {
            best = score;
            selected = k;
            inside = above;
        }
    }
    winner[p] = inside ? selected : -1;
    if (selected < 0) return;
    const unsigned peers = __match_any_sync(lanes, selected);
    const unsigned intersections = __ballot_sync(lanes, inside);
    if (lane == __ffs(peers) - 1) {
        atomicAdd(&areas->assigned[selected], __popc(peers));
        atomicAdd(&areas->intersection[selected], __popc(peers & intersections));
    }
}

__global__ void merge(const Candidates* candidates, const Areas* areas, const int* things,
                      int* ids, Segments* segments) {
    int stuff[classes] = {};
    int count = 0;
    for (int k = 0; k < candidates->count; ++k) {
        ids[k] = 0;
        if (!areas->intersection[k] || !areas->original[k] ||
            double(areas->assigned[k]) / areas->original[k] < 0.8) continue;
        const int label = candidates->label[k];
        if (!things[label] && stuff[label]) {
            ids[k] = stuff[label];
            continue;
        }
        const int id = count + 1;
        if (!things[label]) stuff[label] = id;
        segments->entries[count++] = {id, label, things[label]};
        ids[k] = id;
    }
    segments->count = count;
}

__global__ void write_map(const int* winner, const int* ids, int* map, int pixels) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pixels) map[p] = winner[p] < 0 ? 0 : ids[winner[p]];
}
}  // namespace

void Postprocess::reserve(size_t pixels) {
    candidates_.reserve(1);
    areas_.reserve(1);
    winner_.reserve(pixels);
    ids_.reserve(queries);
    map_.reserve(pixels);
    segments_.reserve(1);
    host_map_.reserve(pixels);
    host_segments_.reserve(1);
}

void Postprocess::run(const float* logits, const float* masks, const int* things,
                      Geometry geometry, cudaStream_t stream) {
    const int pixels = geometry.height * geometry.width;
    const int blocks = (pixels + threads - 1) / threads;
    reserve(pixels);
    check(cudaMemsetAsync(areas_.get(), 0, sizeof(Areas), stream));
    select<<<1, queries, 0, stream>>>(logits, candidates_.get());
    assign<<<blocks, threads, 0, stream>>>(masks, candidates_.get(), geometry, areas_.get(), winner_.get());
    merge<<<1, 1, 0, stream>>>(candidates_.get(), areas_.get(), things, ids_.get(), segments_.get());
    write_map<<<blocks, threads, 0, stream>>>(winner_.get(), ids_.get(), map_.get(), pixels);
    check(cudaGetLastError());
}

void Postprocess::download(Geometry geometry, cudaStream_t stream) {
    check(cudaMemcpyAsync(host_map_.get(), map_.get(), size_t(geometry.height) * geometry.width * sizeof(int),
                          cudaMemcpyDeviceToHost, stream));
    check(cudaMemcpyAsync(host_segments_.get(), segments_.get(), sizeof(Segments), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
}
