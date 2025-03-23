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
constexpr int num_elements_per_thread_decomposed          = 8;
constexpr int num_threads_per_idct_block_decomposed =
    dct_block_size / num_elements_per_thread_decomposed;
constexpr int num_threads_per_thread_block_decomposed =
    num_threads_per_idct_block_decomposed * num_idct_blocks_per_thread_block_decomposed;
constexpr int num_elements_per_thread_block_decomposed =
    num_threads_per_thread_block_decomposed * num_elements_per_thread_decomposed;

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

    const int row_idx        = threadIdx.x % dct_block_dim;
    const int idct_block_idx = threadIdx.x / dct_block_dim;

    // idct block offset
    const int off_in_block = dct_block_size * idct_block_idx;

    int16_t coeffs_local[8];
    *reinterpret_cast<uint4*>(coeffs_local) =
        reinterpret_cast<const uint4*>(coeffs + block_off + off_in_block)[row_idx];

    float vals[8];
    for (int i = 0; i < 8; ++i) {
        vals[i] = coeffs_local[i] * static_cast<float>(qtable_shared[row_idx * 8 + i]);
    }

    idct_vector(vals);

    __shared__ float coeffs_shared[num_elements_per_thread_block_decomposed];
    float* idct_block_coeffs = coeffs_shared + off_in_block;

    for (int i = 0; i < dct_block_dim; ++i) {
        idct_block_coeffs[dct_block_dim * row_idx + i] = vals[i];
    }

    __syncthreads();

    for (int i = 0; i < dct_block_dim; ++i) {
        vals[i] = idct_block_coeffs[dct_block_dim * i + row_idx];
    }

    idct_vector(vals);

    // excessive global access, but transposing in shared memory first, is slower
    const int pixel_idx = block_off + off_in_block;
    for (int i = 0; i < dct_block_dim; ++i) {
        const int idx   = dct_block_dim * i + row_idx;
        const float val = 128 + std::roundf(vals[i]);

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

namespace {

constexpr int num_idct_blocks_per_thread_block_no_shared = 32; // tunable
constexpr int num_elements_per_thread_memory             = 8;
constexpr int num_threads_per_idct_block_memory = dct_block_size / num_elements_per_thread_memory;
constexpr int num_threads_per_thread_block_no_shared =
    num_threads_per_idct_block_memory * num_idct_blocks_per_thread_block_no_shared;

__device__ void transpose(float vals[8])
{
    // https://forums.developer.nvidia.com/t/implement-2d-matrix-transpose-using-warp-shuffle-without-local-memory/208418/10

    float uswap;
    float u0 = vals[0];
    float u1 = vals[1];
    float u2 = vals[2];
    float u3 = vals[3];
    float u4 = vals[4];
    float u5 = vals[5];
    float u6 = vals[6];
    float u7 = vals[7];

    // perform 2x2 movement
    // moving single elements in 2x2 blocks
    // step 1:
    if (!(threadIdx.x & 1)) {
        uswap = u1;
        u1    = u0;
        u0    = uswap;
        uswap = u3;
        u3    = u2;
        u2    = uswap;
        uswap = u5;
        u5    = u4;
        u4    = uswap;
        uswap = u7;
        u7    = u6;
        u6    = uswap;
    }
    // step 2:
    u0 = __shfl_xor_sync(0x000000ff, u0, 1);
    u2 = __shfl_xor_sync(0x000000ff, u2, 1);
    u4 = __shfl_xor_sync(0x000000ff, u4, 1);
    u6 = __shfl_xor_sync(0x000000ff, u6, 1);
    // step 3:
    if (!(threadIdx.x & 1)) {
        uswap = u1;
        u1    = u0;
        u0    = uswap;
        uswap = u3;
        u3    = u2;
        u2    = uswap;
        uswap = u5;
        u5    = u4;
        u4    = uswap;
        uswap = u7;
        u7    = u6;
        u6    = uswap;
    }

    // perform 4x4 movement
    // moving 2x2 elements in 4x4 blocks
    // step 1:
    if (!(threadIdx.x & 2)) {
        uswap = u1;
        u1    = u3;
        u3    = uswap;
        uswap = u0;
        u0    = u2;
        u2    = uswap;
        uswap = u5;
        u5    = u7;
        u7    = uswap;
        uswap = u4;
        u4    = u6;
        u6    = uswap;
    }
    // step 2:
    u0 = __shfl_xor_sync(0x000000ff, u0, 2);
    u1 = __shfl_xor_sync(0x000000ff, u1, 2);
    u4 = __shfl_xor_sync(0x000000ff, u4, 2);
    u5 = __shfl_xor_sync(0x000000ff, u5, 2);
    // step 3:
    if (!(threadIdx.x & 2)) {
        uswap = u1;
        u1    = u3;
        u3    = uswap;
        uswap = u0;
        u0    = u2;
        u2    = uswap;
        uswap = u5;
        u5    = u7;
        u7    = uswap;
        uswap = u4;
        u4    = u6;
        u6    = uswap;
    }
    // perform 8x8 movement
    // moving 4x4 elements in 8x8 blocks
    // step 1:
    if (!(threadIdx.x & 4)) {
        uswap = u0;
        u0    = u4;
        u4    = uswap;
        uswap = u1;
        u1    = u5;
        u5    = uswap;
        uswap = u2;
        u2    = u6;
        u6    = uswap;
        uswap = u3;
        u3    = u7;
        u7    = uswap;
    }
    // step 2:
    u0 = __shfl_xor_sync(0x000000ff, u0, 4);
    u1 = __shfl_xor_sync(0x000000ff, u1, 4);
    u2 = __shfl_xor_sync(0x000000ff, u2, 4);
    u3 = __shfl_xor_sync(0x000000ff, u3, 4);
    // step 3:
    if (!(threadIdx.x & 4)) {
        uswap = u0;
        u0    = u4;
        u4    = uswap;
        uswap = u1;
        u1    = u5;
        u5    = uswap;
        uswap = u2;
        u2    = u6;
        u6    = uswap;
        uswap = u3;
        u3    = u7;
        u7    = uswap;
    }

    vals[0] = u0;
    vals[1] = u1;
    vals[2] = u2;
    vals[3] = u3;
    vals[4] = u4;
    vals[5] = u5;
    vals[6] = u6;
    vals[7] = u7;
}

__global__ void idct_no_shared_kernel(
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

    const int row_idx        = threadIdx.x % dct_block_dim;
    const int idct_block_idx = threadIdx.x / dct_block_dim;

    // idct block offset
    const int off_in_block = dct_block_size * idct_block_idx;

    int16_t coeffs_local[8];
    *reinterpret_cast<uint4*>(coeffs_local) =
        reinterpret_cast<const uint4*>(coeffs + block_off + off_in_block)[row_idx];

    float vals[8];
    for (int i = 0; i < 8; ++i) {
        vals[i] = coeffs_local[i] * static_cast<float>(qtable_shared[row_idx * 8 + i]);
    }

    idct_vector(vals);

    transpose(vals);

    idct_vector(vals);

    transpose(vals);

    uint8_t pixels_regs[8];
    for (int i = 0; i < dct_block_dim; ++i) {
        const float val = 128 + std::roundf(vals[i]);
        pixels_regs[i]  = clampf(val, 0.f, 255.f);
    }
    reinterpret_cast<uint2*>(pixels + block_off + off_in_block)[row_idx] =
        *reinterpret_cast<const uint2*>(pixels_regs);
}

} // namespace

int get_num_idct_blocks_per_thread_block_no_shared()
{
    return num_idct_blocks_per_thread_block_no_shared;
}

bool idct_no_shared(
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

        const int kernel_block_size = num_threads_per_thread_block_no_shared;
        assert(num_blocks_c % num_idct_blocks_per_thread_block_no_shared == 0);
        const int num_kernel_blocks = num_blocks_c / num_idct_blocks_per_thread_block_no_shared;

        idct_no_shared_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            pixels[c].ptr, coeffs[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}

// TODO ideas:
// - have one wavefront
// - mix float and int ops
// - qtable in constant memory
