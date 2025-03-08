// Copyright (c) 2025 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "dct.hpp"
#include "util.hpp"

#include <algorithm>
#include <numbers>

#include <cassert>
#include <stdint.h>

namespace {

void idct_block(uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    // https://en.wikipedia.org/wiki/JPEG#Decoding
    for (int y = 0; y < dct_block_dim; ++y) {
        for (int x = 0; x < dct_block_dim; ++x) {

            float f = 0.f;
            for (int v = 0; v < dct_block_dim; ++v) {

                float ff = 0.f;
                for (int u = 0; u < dct_block_dim; ++u) {
                    const float alpha_u = u == 0 ? 1.f / sqrtf(2.f) : 1;
                    const float cosx =
                        cosf(((2.f * x + 1.f) * u * std::numbers::pi) * (1.f / 16.f));

                    const int16_t qval = qtable[dct_block_dim * v + u];

                    ff += alpha_u * coeffs[dct_block_dim * v + u] * cosx * qval;
                }

                const float alpha_v = v == 0 ? 1.f / sqrtf(2.f) : 1;
                const float cosy    = cosf(((2.f * y + 1.f) * v * std::numbers::pi) * (1.f / 16.f));
                f += ff * alpha_v * cosy;
            }

            const float val = 128 + std::roundf(.25f * f);

            pixels[8 * y + x] = std::clamp(val, 0.f, 255.f);
        }
    }
}

} // namespace

void idct_cpu(
    std::vector<std::vector<uint8_t>>& pixels,
    const std::vector<std::vector<int16_t>>& coeffs,
    const std::vector<std::vector<uint16_t>>& qtable,
    const std::vector<int>& num_blocks)
{
    assert(pixels.size() == coeffs.size());
    assert(coeffs.size() == qtable.size());
    assert(qtable.size() == num_blocks.size());
    const int num_components = pixels.size();

    for (int c = 0; c < num_components; ++c) {
        const int num_blocks_c = num_blocks[c];
        for (int i = 0; i < num_blocks_c; ++i) {
            const int off = dct_block_size * i;
            idct_block(
                pixels[c].data() + off, coeffs[c].data() + off, qtable[c].data(), num_blocks_c);
        }
    }
}

namespace {

// [print(f'{(0.5 / math.sqrt(2) if j == 0 else 0.5) * math.cos((2 * i + 1) * j * math.pi / 16):+.9f}{", " if i < 7 else ",\n"}', end="") for j in range(8) for i in range(8)]
// clang-format off
constexpr float lut[dct_block_size] = {
    +0.353553391, +0.353553391, +0.353553391, +0.353553391, +0.353553391, +0.353553391, +0.353553391, +0.353553391,
    +0.490392640, +0.415734806, +0.277785117, +0.097545161, -0.097545161, -0.277785117, -0.415734806, -0.490392640,
    +0.461939766, +0.191341716, -0.191341716, -0.461939766, -0.461939766, -0.191341716, +0.191341716, +0.461939766,
    +0.415734806, -0.097545161, -0.490392640, -0.277785117, +0.277785117, +0.490392640, +0.097545161, -0.415734806,
    +0.353553391, -0.353553391, -0.353553391, +0.353553391, +0.353553391, -0.353553391, -0.353553391, +0.353553391,
    +0.277785117, -0.490392640, +0.097545161, +0.415734806, -0.415734806, -0.097545161, +0.490392640, -0.277785117,
    +0.191341716, -0.461939766, +0.461939766, -0.191341716, -0.191341716, +0.461939766, -0.461939766, +0.191341716,
    +0.097545161, -0.277785117, +0.415734806, -0.490392640, +0.490392640, -0.415734806, +0.277785117, -0.097545161,
};
__device__ constexpr float lut_gpu[dct_block_size] = {
    +0.353553391, +0.353553391, +0.353553391, +0.353553391, +0.353553391, +0.353553391, +0.353553391, +0.353553391,
    +0.490392640, +0.415734806, +0.277785117, +0.097545161, -0.097545161, -0.277785117, -0.415734806, -0.490392640,
    +0.461939766, +0.191341716, -0.191341716, -0.461939766, -0.461939766, -0.191341716, +0.191341716, +0.461939766,
    +0.415734806, -0.097545161, -0.490392640, -0.277785117, +0.277785117, +0.490392640, +0.097545161, -0.415734806,
    +0.353553391, -0.353553391, -0.353553391, +0.353553391, +0.353553391, -0.353553391, -0.353553391, +0.353553391,
    +0.277785117, -0.490392640, +0.097545161, +0.415734806, -0.415734806, -0.097545161, +0.490392640, -0.277785117,
    +0.191341716, -0.461939766, +0.461939766, -0.191341716, -0.191341716, +0.461939766, -0.461939766, +0.191341716,
    +0.097545161, -0.277785117, +0.415734806, -0.490392640, +0.490392640, -0.415734806, +0.277785117, -0.097545161,
};
// clang-format on

void idct_block_lut(uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    // https://en.wikipedia.org/wiki/JPEG#Decoding
    for (int y = 0; y < dct_block_dim; ++y) {
        for (int x = 0; x < dct_block_dim; ++x) {

            float f = 0.f;
            for (int v = 0; v < dct_block_dim; ++v) {

                float ff = 0.f;
                for (int u = 0; u < dct_block_dim; ++u) {
                    const int16_t qval = qtable[dct_block_dim * v + u];
                    ff += coeffs[dct_block_dim * v + u] * lut[dct_block_dim * u + x] * qval;
                }
                f += ff * lut[dct_block_dim * v + y];
            }

            const float val = 128 + std::roundf(f);

            pixels[dct_block_dim * y + x] = std::clamp(val, 0.f, 255.f);
        }
    }
}

} // namespace

void idct_cpu_lut(
    std::vector<std::vector<uint8_t>>& pixels,
    const std::vector<std::vector<int16_t>>& coeffs,
    const std::vector<std::vector<uint16_t>>& qtable,
    const std::vector<int>& num_blocks)
{
    assert(pixels.size() == coeffs.size());
    assert(coeffs.size() == qtable.size());
    assert(qtable.size() == num_blocks.size());
    const int num_components = pixels.size();

    for (int c = 0; c < num_components; ++c) {
        const int num_blocks_c = num_blocks[c];
        for (int i = 0; i < num_blocks_c; ++i) {
            const int off = dct_block_size * i;
            idct_block_lut(
                pixels[c].data() + off, coeffs[c].data() + off, qtable[c].data(), num_blocks_c);
        }
    }
}

namespace {

__device__ float clampf(float x, float min, float max) { return fmaxf(min, fminf(x, max)); }

// 64 values in one idct
constexpr int idct_blocks_per_thread_block  = 4;
constexpr int num_elements_per_thread_block = idct_blocks_per_thread_block * dct_block_size;

// naive version, every thread handles one value
__global__ void idct_kernel(
    uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    assert(blockDim.x == num_elements_per_thread_block);
    const int block_off = num_elements_per_thread_block * blockIdx.x;
    const int pixel_idx = block_off + threadIdx.x;

    // do not early exit partial blocks because all threads must participate in load and sync

    __shared__ int16_t coeffs_shared[num_elements_per_thread_block];
    for (int i = threadIdx.x; i < num_elements_per_thread_block; i += blockDim.x) {
        const int load_idx = block_off + i;
        if (load_idx < num_blocks * dct_block_size) {
            coeffs_shared[i] = coeffs[load_idx];
        } else {
            // always initialize all values
            coeffs_shared[i] = 0;
        }
    }

    __shared__ uint16_t qtable_shared[dct_block_size];
    assert(blockDim.x >= dct_block_size);
    if (threadIdx.x < dct_block_size) {
        qtable_shared[threadIdx.x] = qtable[threadIdx.x];
    }

    __syncthreads();

    if (pixel_idx >= num_blocks * dct_block_size) {
        return;
    }

    const int idx_of_block = threadIdx.x / dct_block_size;
    const int idx_in_block = threadIdx.x % dct_block_size;
    const int y            = idx_in_block / dct_block_dim;
    const int x            = idx_in_block % dct_block_dim;

    float f = 0.f;
    for (int v = 0; v < dct_block_dim; ++v) {

        float ff = 0.f;
        for (int u = 0; u < dct_block_dim; ++u) {
            const float alpha_u = u == 0 ? 1.f / sqrtf(2.f) : 1;
            const float cosx    = cosf(((2.f * x + 1.f) * u * std::numbers::pi) * (1.f / 16.f));

            const int16_t qval = qtable_shared[dct_block_dim * v + u];

            const size_t idx_in = dct_block_size * idx_of_block + dct_block_dim * v + u;
            ff += alpha_u * coeffs_shared[idx_in] * cosx * qval;
        }

        const float alpha_v = v == 0 ? 1.f / sqrtf(2.f) : 1;
        const float cosy    = cosf(((2.f * y + 1.f) * v * std::numbers::pi) * (1.f / 16.f));
        f += ff * alpha_v * cosy;
    }

    const float val = 128 + std::roundf(.25f * f);

    pixels[pixel_idx] = clampf(val, 0.f, 255.f);
}

} // namespace

bool idct(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream)
{
    assert(pixels.size() == coeffs.size());
    assert(coeffs.size() == qtable.size());
    assert(qtable.size() == num_blocks.size());
    const int num_components = pixels.size();

    for (int c = 0; c < num_components; ++c) {
        const int num_blocks_c = num_blocks[c];

        const int kernel_block_size = num_elements_per_thread_block;
        const int num_kernel_blocks =
            (num_blocks_c + idct_blocks_per_thread_block - 1) / idct_blocks_per_thread_block;

        idct_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            pixels[c].ptr, coeffs[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}

namespace {

__global__ void idct_lut_kernel(
    uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    assert(blockDim.x == num_elements_per_thread_block);
    const int block_off = num_elements_per_thread_block * blockIdx.x;
    const int pixel_idx = block_off + threadIdx.x;

    // do not early exit partial blocks because all threads must participate in load and sync

    __shared__ int16_t coeffs_shared[num_elements_per_thread_block];
    for (int i = threadIdx.x; i < num_elements_per_thread_block; i += blockDim.x) {
        const int load_idx = block_off + i;
        if (load_idx < num_blocks * dct_block_size) {
            coeffs_shared[i] = coeffs[load_idx];
        } else {
            // always initialize all values
            coeffs_shared[i] = 0;
        }
    }

    __shared__ uint16_t qtable_shared[dct_block_size];
    assert(blockDim.x >= dct_block_size);
    if (threadIdx.x < dct_block_size) {
        qtable_shared[threadIdx.x] = qtable[threadIdx.x];
    }

    __syncthreads();

    if (pixel_idx >= num_blocks * dct_block_size) {
        return;
    }

    const int idx_of_block = threadIdx.x / dct_block_size;
    const int idx_in_block = threadIdx.x % dct_block_size;
    const int y            = idx_in_block / dct_block_dim;
    const int x            = idx_in_block % dct_block_dim;

    float f = 0.f;
    for (int v = 0; v < dct_block_dim; ++v) {

        float ff = 0.f;
        for (int u = 0; u < dct_block_dim; ++u) {
            const int16_t qval = qtable_shared[dct_block_dim * v + u];

            const size_t idx_in = dct_block_size * idx_of_block + dct_block_dim * v + u;
            ff += coeffs_shared[idx_in] * lut_gpu[dct_block_dim * u + x] * qval;
        }

        f += ff * lut_gpu[dct_block_dim * v + y];
    }

    const float val = 128 + std::roundf(f);

    pixels[pixel_idx] = clampf(val, 0.f, 255.f);
}

} // namespace

bool idct_lut(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream)
{
    assert(pixels.size() == coeffs.size());
    assert(coeffs.size() == qtable.size());
    assert(qtable.size() == num_blocks.size());
    const int num_components = pixels.size();

    for (int c = 0; c < num_components; ++c) {
        const int num_blocks_c = num_blocks[c];

        const int kernel_block_size = num_elements_per_thread_block;
        const int num_kernel_blocks =
            (num_blocks_c + idct_blocks_per_thread_block - 1) / idct_blocks_per_thread_block;

        idct_lut_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            pixels[c].ptr, coeffs[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}
