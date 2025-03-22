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

__device__ float clampf(float x, float min, float max) { return fmaxf(min, fminf(x, max)); }

constexpr int num_elements_per_thread_block_naive =
    num_idct_blocks_per_thread_block_naive * dct_block_size;

// naive version, every thread handles one value
__global__ void idct_kernel(
    uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    assert(blockDim.x == num_elements_per_thread_block_naive);
    const int block_off = num_elements_per_thread_block_naive * blockIdx.x;
    const int pixel_idx = block_off + threadIdx.x;

    __shared__ int16_t coeffs_shared[num_elements_per_thread_block_naive];
    for (int i = threadIdx.x; i < num_elements_per_thread_block_naive; i += blockDim.x) {
        const int load_idx = block_off + i;
        coeffs_shared[i]   = coeffs[load_idx];
    }

    __shared__ uint16_t qtable_shared[dct_block_size];
    assert(blockDim.x >= dct_block_size);
    if (threadIdx.x < dct_block_size) {
        qtable_shared[threadIdx.x] = qtable[threadIdx.x];
    }

    __syncthreads();

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

bool idct_naive(
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

        const int kernel_block_size = num_elements_per_thread_block_naive;
        assert(num_blocks_c % num_idct_blocks_per_thread_block_naive == 0);
        const int num_kernel_blocks = num_blocks_c / num_idct_blocks_per_thread_block_naive;

        idct_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            pixels[c].ptr, coeffs[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}

namespace {

constexpr int num_elements_per_thread_block_lut =
    num_idct_blocks_per_thread_block_lut * dct_block_size;

__global__ void idct_lut_kernel(
    uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    assert(blockDim.x == num_elements_per_thread_block_lut);
    const int block_off = num_elements_per_thread_block_lut * blockIdx.x;
    const int pixel_idx = block_off + threadIdx.x;

    __shared__ int16_t coeffs_shared[num_elements_per_thread_block_lut];
    for (int i = threadIdx.x; i < num_elements_per_thread_block_lut; i += blockDim.x) {
        const int load_idx = block_off + i;
        coeffs_shared[i]   = coeffs[load_idx];
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

        const int kernel_block_size = num_elements_per_thread_block_lut;
        assert(num_blocks_c % num_idct_blocks_per_thread_block_lut == 0);
        const int num_kernel_blocks = num_blocks_c / num_idct_blocks_per_thread_block_lut;

        idct_lut_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            pixels[c].ptr, coeffs[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}

namespace {

constexpr int num_elements_per_thread_block_seperable =
    num_idct_blocks_per_thread_block_seperable * dct_block_size;

__global__ void idct_seperable_kernel(
    uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    assert(blockDim.x == num_elements_per_thread_block_seperable);
    const int block_off = num_elements_per_thread_block_seperable * blockIdx.x;
    const int pixel_idx = block_off + threadIdx.x;

    __shared__ int16_t coeffs_shared[num_elements_per_thread_block_seperable];
    for (int i = threadIdx.x; i < num_elements_per_thread_block_seperable; i += blockDim.x) {
        const int load_idx = block_off + i;
        coeffs_shared[i]   = coeffs[load_idx];
    }

    __shared__ uint16_t qtable_shared[dct_block_size];
    assert(blockDim.x >= dct_block_size);
    if (threadIdx.x < dct_block_size) {
        qtable_shared[threadIdx.x] = qtable[threadIdx.x];
    }

    __syncthreads();

    const int idx_of_block = threadIdx.x / dct_block_size;
    const int idx_in_block = threadIdx.x % dct_block_size;
    const int y            = idx_in_block / dct_block_dim;
    const int x            = idx_in_block % dct_block_dim;

    float f;
    __shared__ float coeffs_intermediate[num_elements_per_thread_block_seperable];

    // perform the inner loop of `idct_lut_kernel` with v = y
    f = 0.f;
    for (int u = 0; u < dct_block_dim; ++u) {
        const int16_t qval = qtable_shared[dct_block_dim * y + u];

        const size_t idx_in = dct_block_size * idx_of_block + dct_block_dim * y + u;
        f += coeffs_shared[idx_in] * lut_gpu[dct_block_dim * u + x] * qval;
    }
    coeffs_intermediate[dct_block_size * idx_of_block + dct_block_dim * y + x] = f;
    __syncthreads();
    // perform the outer loop of `idct_lut_kernel`
    f = 0.f;
    for (int v = 0; v < dct_block_dim; ++v) {
        const size_t idx_in = dct_block_size * idx_of_block + dct_block_dim * v + x;
        const float ff      = coeffs_intermediate[idx_in];

        f += ff * lut_gpu[dct_block_dim * v + y];
    }

    const float val = 128 + std::roundf(f);

    pixels[pixel_idx] = clampf(val, 0.f, 255.f);
}

} // namespace

bool idct_seperable(
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

        const int kernel_block_size = num_elements_per_thread_block_seperable;
        assert(num_blocks_c % num_idct_blocks_per_thread_block_seperable == 0);
        const int num_kernel_blocks = num_blocks_c / num_idct_blocks_per_thread_block_seperable;

        idct_seperable_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            pixels[c].ptr, coeffs[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}

namespace {

constexpr int num_elements_per_thread_block_memory =
    num_idct_blocks_per_thread_block_memory * dct_block_size;

__global__ void idct_memory_kernel(
    uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    assert(blockDim.x == num_elements_per_thread_block_memory);
    const int block_off = num_elements_per_thread_block_memory * blockIdx.x;

    __shared__ int16_t coeffs_shared[num_elements_per_thread_block_memory];
    for (int i = threadIdx.x; i < num_elements_per_thread_block_memory / 2; i += blockDim.x) {
        reinterpret_cast<uint32_t*>(coeffs_shared)[i] =
            reinterpret_cast<const uint32_t*>(coeffs + block_off)[i];
    }

    __shared__ uint16_t qtable_shared[dct_block_size];
    assert(blockDim.x >= dct_block_size);
    if (threadIdx.x < dct_block_size) {
        qtable_shared[threadIdx.x] = qtable[threadIdx.x];
    }

    __syncthreads();

    const int idx_of_block = threadIdx.x / dct_block_size;
    const int idx_in_block = threadIdx.x % dct_block_size;
    const int y            = idx_in_block / dct_block_dim;
    const int x            = idx_in_block % dct_block_dim;

    float f;
    __shared__ float coeffs_intermediate[num_elements_per_thread_block_memory];

    // perform the inner loop of `idct_lut_kernel` with v = y
    f = 0.f;
    for (int u = 0; u < dct_block_dim; ++u) {
        const int16_t qval = qtable_shared[dct_block_dim * y + u];

        const size_t idx_in = dct_block_size * idx_of_block + dct_block_dim * y + u;
        f += coeffs_shared[idx_in] * lut_gpu[dct_block_dim * u + x] * qval;
    }
    coeffs_intermediate[dct_block_size * idx_of_block + dct_block_dim * y + x] = f;
    __syncthreads();
    // perform the outer loop of `idct_lut_kernel`
    f = 0.f;
    for (int v = 0; v < dct_block_dim; ++v) {
        const size_t idx_in = dct_block_size * idx_of_block + dct_block_dim * v + x;
        const float ff      = coeffs_intermediate[idx_in];

        f += ff * lut_gpu[dct_block_dim * v + y];
    }

    __shared__ uint8_t shared_out[num_elements_per_thread_block_memory];

    const float val = 128 + std::roundf(f);

    shared_out[dct_block_size * idx_of_block + dct_block_dim * y + x] = clampf(val, 0.f, 255.f);

    __syncthreads();
    for (int i = threadIdx.x; i < num_elements_per_thread_block_memory / 4; i += blockDim.x) {
        reinterpret_cast<uint32_t*>(pixels + block_off)[i] =
            reinterpret_cast<uint32_t*>(shared_out)[i];
    }
}

} // namespace

bool idct_memory(
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

        const int kernel_block_size = num_elements_per_thread_block_memory;
        assert(num_blocks_c % num_idct_blocks_per_thread_block_memory == 0);
        const int num_kernel_blocks = num_blocks_c / num_idct_blocks_per_thread_block_memory;

        idct_memory_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            pixels[c].ptr, coeffs[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}
