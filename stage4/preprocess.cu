#include "preprocess.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <setjmp.h>
#include <vector>

#include <jpeglib.h>

namespace {
constexpr int coefficient_bits = 22;
constexpr int coefficient_scale = 1 << coefficient_bits;

struct JpegError {
    jpeg_error_mgr callbacks;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
};

void jpeg_error(j_common_ptr info) {
    auto* error = reinterpret_cast<JpegError*>(info->err);
    info->err->format_message(info, error->message);
    longjmp(error->jump, 1);
}

struct Decoder {
    jpeg_decompress_struct info{};
    JpegError error{};
    ~Decoder() { jpeg_destroy_decompress(&info); }
};

int orientation(const jpeg_decompress_struct& info) {
    for (auto* marker = info.marker_list; marker; marker = marker->next) {
        if (marker->data_length < 14 || std::memcmp(marker->data, "Exif\0\0", 6)) continue;
        const auto* tiff = marker->data + 6;
        const size_t length = marker->data_length - 6;
        const bool little = tiff[0] == 'I' && tiff[1] == 'I';
        if (!little && !(tiff[0] == 'M' && tiff[1] == 'M')) continue;
        auto read = [&](size_t offset, int bytes) {
            uint32_t value = 0;
            for (int i = 0; i < bytes; ++i)
                value = (value << 8) | tiff[offset + (little ? bytes - 1 - i : i)];
            return value;
        };
        if (read(2, 2) != 42) continue;
        size_t offset = read(4, 4);
        if (offset > length - 2) continue;
        auto count = read(offset, 2);
        offset += 2;
        if (count > (length - offset) / 12) continue;
        for (uint32_t i = 0; i < count; ++i, offset += 12) {
            if (read(offset, 2) == 0x112 && read(offset + 2, 2) == 3 && read(offset + 4, 4) == 1) {
                auto value = read(offset + 8, 2);
                return value >= 1 && value <= 8 ? value : 1;
            }
        }
    }
    return 1;
}

Geometry decode(const unsigned char* jpeg, size_t length, Buffer<unsigned char, true>& rgb) {
    auto decoder = std::make_unique<Decoder>();
    auto& info = decoder->info;
    info.err = jpeg_std_error(&decoder->error.callbacks);
    info.err->error_exit = jpeg_error;
    if (setjmp(decoder->error.jump)) throw std::runtime_error(decoder->error.message);

    jpeg_create_decompress(&info);
    jpeg_save_markers(&info, JPEG_APP0 + 1, 0xffff);
    jpeg_mem_src(&info, jpeg, length);
    jpeg_read_header(&info, TRUE);
    const int transform = orientation(info);
    const bool cmyk = info.jpeg_color_space == JCS_CMYK || info.jpeg_color_space == JCS_YCCK;
    info.out_color_space = cmyk ? JCS_CMYK : JCS_RGB;
    jpeg_start_decompress(&info);
    const int h = info.output_height, w = info.output_width;
    const int height = transform >= 5 ? w : h;
    const int width = transform >= 5 ? h : w;
    const size_t bytes = size_t(height) * width * 3;
    rgb.reserve(bytes + size_t(w) * info.output_components);
    auto* row = rgb.get() + bytes;
    while (info.output_scanline < info.output_height) {
        const int y = info.output_scanline;
        auto* scanline = transform == 1 && !cmyk ? rgb.get() + size_t(y) * w * 3 : row;
        jpeg_read_scanlines(&info, &scanline, 1);
        if (scanline != row) continue;
        for (int x = 0; x < w; ++x) {
            int dx = x, dy = y;
            switch (transform) {
                case 2: dx = w - 1 - x; break;
                case 3: dx = w - 1 - x; dy = h - 1 - y; break;
                case 4: dy = h - 1 - y; break;
                case 5: dx = y; dy = x; break;
                case 6: dx = h - 1 - y; dy = x; break;
                case 7: dx = h - 1 - y; dy = w - 1 - x; break;
                case 8: dx = y; dy = w - 1 - x; break;
            }
            auto* pixel = rgb.get() + (size_t(dy) * width + dx) * 3;
            if (cmyk) {
                for (int c = 0; c < 3; ++c) pixel[c] = (row[x * 4 + c] * row[x * 4 + 3] + 127) / 255;
            } else {
                std::memcpy(pixel, row + x * 3, 3);
            }
        }
    }
    jpeg_finish_decompress(&info);
    const double scale = double(side) / std::max(height, width);
    return {height, width, std::max(1, int(std::nearbyint(height * scale))),
            std::max(1, int(std::nearbyint(width * scale)))};
}

// Pixel-center triangular filter with downsampling support and per-pass uint8 rounding.
void build_axis(int input, int output, AxisTap* taps, std::vector<int>& coefficients) {
    const double scale = double(input) / output;
    const double support = std::max(1.0, scale);
    const double inverse_support = 1.0 / support;
    for (int i = 0; i < output; ++i) {
        const double center = (i + 0.5) * scale;
        const int first = std::max(0, int(center - support + 0.5));
        const int last = std::min(input, int(center + support + 0.5));
        taps[i] = {first, last - first, int(coefficients.size())};
        auto weight = [&](int j) { return std::max(0.0, 1 - std::abs((j - center + 0.5) * inverse_support)); };
        double sum = 0;
        for (int j = first; j < last; ++j) sum += weight(j);
        for (int j = first; j < last; ++j)
            coefficients.push_back(int(weight(j) / sum * coefficient_scale + 0.5));
    }
}

__global__ void horizontal(const unsigned char* rgb, unsigned char* intermediate, int w, int h,
                           int valid_w, const AxisTap* taps, const int* coefficients) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= h * valid_w * 3) return;
    const int x = index / 3 % valid_w, y = index / (3 * valid_w), c = index % 3;
    const auto tap = taps[x];
    int sum = coefficient_scale / 2;
    for (int i = 0; i < tap.count; ++i)
        sum += rgb[(y * w + tap.first + i) * 3 + c] * coefficients[tap.offset + i];
    intermediate[index] = min(255, sum >> coefficient_bits);
}

__global__ void vertical(const unsigned char* intermediate, float* image, int valid_w, int valid_h,
                         const AxisTap* taps, const int* coefficients) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= side * side) return;
    const int x = index % side, y = index / side;
    constexpr float mean[] = {123.675f, 116.280f, 103.530f};
    constexpr float stddev[] = {58.395f, 57.120f, 57.375f};
    for (int c = 0; c < 3; ++c) {
        int pixel = 128;
        if (x < valid_w && y < valid_h) {
            const auto tap = taps[y];
            int sum = coefficient_scale / 2;
            for (int i = 0; i < tap.count; ++i)
                sum += intermediate[((tap.first + i) * valid_w + x) * 3 + c] * coefficients[tap.offset + i];
            pixel = min(255, sum >> coefficient_bits);
        }
        image[c * side * side + index] = (pixel - mean[c]) / stddev[c];
    }
}
}  // namespace

Geometry Preprocess::run(const unsigned char* jpeg, size_t length, float* image, cudaStream_t stream) {
    Geometry geometry;
    {
        Range range("decode");
        geometry = decode(jpeg, length, rgb_);
    }
    const auto [h, w, valid_h, valid_w] = geometry;
    const size_t rgb_bytes = size_t(h) * w * 3;
    const size_t intermediate_bytes = size_t(h) * valid_w * 3;
    std::vector<int> weights;
    {
        Range range("preprocess");
        taps_.reserve(valid_w + valid_h);
        build_axis(w, valid_w, taps_.get(), weights);
        build_axis(h, valid_h, taps_.get() + valid_w, weights);
        coefficients_.reserve(weights.size());
        std::copy(weights.begin(), weights.end(), coefficients_.get());
        device_rgb_.reserve(rgb_bytes);
        intermediate_.reserve(intermediate_bytes);
        device_taps_.reserve(valid_w + valid_h);
        device_coefficients_.reserve(weights.size());
    }
    {
        Range range("H2D");
        check(cudaMemcpyAsync(device_rgb_.get(), rgb_.get(), rgb_bytes, cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(device_taps_.get(), taps_.get(), (valid_w + valid_h) * sizeof(AxisTap), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(device_coefficients_.get(), coefficients_.get(), weights.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    {
        Range range("preprocess_gpu");
        horizontal<<<(intermediate_bytes + 255) / 256, 256, 0, stream>>>(
            device_rgb_.get(), intermediate_.get(), w, h, valid_w, device_taps_.get(), device_coefficients_.get());
        vertical<<<side * side / 256, 256, 0, stream>>>(
            intermediate_.get(), image, valid_w, valid_h, device_taps_.get() + valid_w, device_coefficients_.get());
        check(cudaGetLastError());
    }
    return geometry;
}
