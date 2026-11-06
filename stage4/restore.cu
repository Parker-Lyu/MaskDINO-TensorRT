#include "postprocess.cuh"

namespace {
struct Coordinate {
    int left;
    int right;
    float fraction;
};

__device__ Coordinate source(int target, int input_size, float scale) {
    const float position = fmaxf(0, fmaf(scale, target + 0.5f, -0.5f));
    const int left = int(position);
    return {left, min(left + 1, input_size - 1), position - left};
}

__device__ float bilinear(float a, float b, float c, float d, float x, float y) {
    return fmaf(1 - y, fmaf(1 - x, a, x * b), y * fmaf(1 - x, c, x * d));
}

__device__ float square_mask(const float* mask, int x, int y) {
    const auto sx = source(x, mask_side, 0.25f);
    const auto sy = source(y, mask_side, 0.25f);
    return bilinear(mask[sy.left * mask_side + sx.left], mask[sy.left * mask_side + sx.right],
                    mask[sy.right * mask_side + sx.left], mask[sy.right * mask_side + sx.right],
                    sx.fraction, sy.fraction);
}
}  // namespace

__device__ float restored_mask(const float* mask, Geometry geometry, int x, int y) {
    const auto sx = source(x, geometry.valid_width, float(geometry.valid_width) / geometry.width);
    const auto sy = source(y, geometry.valid_height, float(geometry.valid_height) / geometry.height);
    if (geometry.valid_width == geometry.width && geometry.valid_height == geometry.height)
        return square_mask(mask, x, y);
    return bilinear(square_mask(mask, sx.left, sy.left), square_mask(mask, sx.right, sy.left),
                    square_mask(mask, sx.left, sy.right), square_mask(mask, sx.right, sy.right),
                    sx.fraction, sy.fraction);
}
