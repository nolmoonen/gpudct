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

constexpr int num_threads_per_dct_block_naive        = dct_block_size;
constexpr int num_idct_blocks_per_thread_block_naive = 4; // tunable
constexpr int num_elements_per_thread_block_naive =
    num_idct_blocks_per_thread_block_naive * num_threads_per_dct_block_naive;

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

int get_num_idct_blocks_per_thread_block_naive() { return num_idct_blocks_per_thread_block_naive; }

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

constexpr int num_threads_per_dct_block_lut        = dct_block_size;
constexpr int num_idct_blocks_per_thread_block_lut = 4; // tunable
constexpr int num_elements_per_thread_block_lut =
    num_idct_blocks_per_thread_block_lut * num_threads_per_dct_block_lut;

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

int get_num_idct_blocks_per_thread_block_lut() { return num_idct_blocks_per_thread_block_lut; }

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

constexpr int num_threads_per_dct_block_seperable        = dct_block_size;
constexpr int num_idct_blocks_per_thread_block_seperable = 4; // tunable
constexpr int num_elements_per_thread_block_seperable =
    num_idct_blocks_per_thread_block_seperable * num_threads_per_dct_block_seperable;

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

int get_num_idct_blocks_per_thread_block_seperable()
{
    return num_idct_blocks_per_thread_block_seperable;
}

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

// TODO(nol) have one kernel in between to demonstrate that doing dequantization when loading
//   decreases the PSNR? may not be needed when having a more stable PSNR measure

namespace {

constexpr int num_idct_blocks_per_thread_block_decomposed = 32; // tunable
constexpr int num_elements_per_thread                     = 8;
constexpr int num_threads_per_idct_block = dct_block_size / num_elements_per_thread;
constexpr int num_threads_per_thread_block_decomposed =
    num_threads_per_idct_block * num_idct_blocks_per_thread_block_decomposed;
constexpr int num_elements_per_thread_block_decomposed =
    num_threads_per_thread_block_decomposed * num_elements_per_thread;

// High-Efficiency and Low-Power Architectures for 2-D DCT and IDCT Based on CORDIC Rotation, Sung et al. 2006
__device__ void idct_vector(float vals[8])
{
    constexpr float sung_a = 1.3870398453221475; // print(f'{math.sqrt(2)*math.cos(1*math.pi/16)}')
    constexpr float sung_b = 1.3065629648763766; // print(f'{math.sqrt(2)*math.cos(2*math.pi/16)}')
    constexpr float sung_c = 1.1758756024193588; // print(f'{math.sqrt(2)*math.cos(3*math.pi/16)}')
    constexpr float sung_d = 0.7856949583871023; // print(f'{math.sqrt(2)*math.cos(5*math.pi/16)}')
    constexpr float sung_e = 0.5411961001461971; // print(f'{math.sqrt(2)*math.cos(6*math.pi/16)}')
    constexpr float sung_f = 0.2758993792829431; // print(f'{math.sqrt(2)*math.cos(7*math.pi/16)}')
    constexpr float sung_t = 0.3535533905932737; // print(f'{1/math.sqrt(8)}')

    const float y0 = vals[0];
    const float y1 = vals[1];
    const float y2 = vals[2];
    const float y3 = vals[3];
    const float y4 = vals[4];
    const float y5 = vals[5];
    const float y6 = vals[6];
    const float y7 = vals[7];

    const float yg = (y0 + y4) + (y2 * sung_b + y6 * sung_e);
    const float yh = y7 * sung_f + y1 * sung_a + y3 * sung_c + y5 * sung_d;
    const float yi = (y0 + y4) - (y2 * sung_b + y6 * sung_e);
    const float yj = y7 * sung_a - y1 * sung_f + y3 * sung_d - y5 * sung_c;

    const float yk = (y0 - y4) + (y2 * sung_e - y6 * sung_b);
    const float yl = y1 * sung_c - y7 * sung_d - y3 * sung_f - y5 * sung_a;
    const float ym = (y0 - y4) - (y2 * sung_e - y6 * sung_b);
    const float yn = y1 * sung_d + y7 * sung_c - y3 * sung_a + y5 * sung_f;

    vals[0] = sung_t * (yg + yh);
    vals[7] = sung_t * (yg - yh);
    vals[4] = sung_t * (yi + yj);
    vals[3] = sung_t * (yi - yj);

    vals[1] = sung_t * (yk + yl);
    vals[5] = sung_t * (ym - yn);
    vals[2] = sung_t * (ym + yn);
    vals[6] = sung_t * (yk - yl);
}

__global__ void idct_decomposed_kernel(
    uint8_t* pixels, const int16_t* coeffs, const uint16_t* qtable, int num_blocks)
{
    assert(blockDim.x == num_threads_per_thread_block_decomposed);
    const int block_off = num_elements_per_thread_block_decomposed * blockIdx.x;

    __shared__ uint16_t qtable_shared[dct_block_size];
    assert(blockDim.x >= dct_block_size);
    if (threadIdx.x < dct_block_size) {
        qtable_shared[threadIdx.x] = qtable[threadIdx.x];
    }

    __syncthreads();

    __shared__ float coeffs_shared[num_elements_per_thread_block_decomposed];
    for (int i = threadIdx.x; i < num_elements_per_thread_block_decomposed; i += blockDim.x) {
        const int load_idx = block_off + i;
        coeffs_shared[i] = coeffs[load_idx] * static_cast<float>(qtable_shared[i % dct_block_size]);
    }
    __syncthreads();

    const int row_idx        = threadIdx.x % dct_block_dim;
    const int idct_block_idx = threadIdx.x / dct_block_dim;

    const int off_in_block = dct_block_size * idct_block_idx;

    float* idct_block_coeffs = coeffs_shared + off_in_block;

    float vals[8];

    for (int i = 0; i < dct_block_dim; ++i) {
        vals[i] = idct_block_coeffs[dct_block_dim * row_idx + i];
    }
    idct_vector(vals);
    for (int i = 0; i < dct_block_dim; ++i) {
        idct_block_coeffs[dct_block_dim * row_idx + i] = vals[i];
    }
    __syncthreads();
    for (int i = 0; i < dct_block_dim; ++i) {
        vals[i] = idct_block_coeffs[dct_block_dim * i + row_idx];
    }
    idct_vector(vals);
    for (int i = 0; i < dct_block_dim; ++i) {
        idct_block_coeffs[dct_block_dim * i + row_idx] = vals[i];
    }
    __syncthreads();

    const int pixel_idx = block_off + off_in_block;
    for (int i = 0; i < dct_block_dim; ++i) {
        const int idx = dct_block_dim * row_idx + i;

        const float val = 128 + std::roundf(idct_block_coeffs[idx]);

        pixels[pixel_idx + idx] = clampf(val, 0.f, 255.f);
    }
}

} // namespace

int get_num_idct_blocks_per_thread_block_decomposed()
{
    return num_idct_blocks_per_thread_block_decomposed;
}

bool idct_decomposed(
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

        const int kernel_block_size = num_threads_per_thread_block_decomposed;
        assert(num_blocks_c % num_idct_blocks_per_thread_block_decomposed == 0);
        const int num_kernel_blocks = num_blocks_c / num_idct_blocks_per_thread_block_decomposed;

        idct_decomposed_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            pixels[c].ptr, coeffs[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}
