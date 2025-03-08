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

#include "psnr.hpp"
#include "util.hpp"

#include <cub/device/device_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <stdint.h>

namespace {

constexpr int warp_size = 32;
static_assert(dct_block_size % warp_size == 0);
constexpr int elements_per_thread = dct_block_size / warp_size;

constexpr int warps_per_thread_block = 8; // configurable
constexpr int thread_block_size      = warps_per_thread_block * warp_size;

__global__ void calc_max_squared_diff(
    int* max_squared_diff_per_block,
    const uint8_t* pixels_a,
    const uint8_t* pixels_b,
    int num_blocks)
{
    assert(blockDim.x == thread_block_size);
    const int tid = thread_block_size * blockIdx.x + threadIdx.x;

    if (elements_per_thread * tid >= num_blocks * dct_block_size) {
        return;
    }

    int sum_tid = 0;
    for (int i = 0; i < elements_per_thread; ++i) {
        const int diff_i =
            pixels_a[elements_per_thread * tid + i] - pixels_b[elements_per_thread * tid + i];
        sum_tid += diff_i * diff_i;
    }

    using warp_reduce = cub::WarpReduce<int>;

    __shared__ typename warp_reduce::TempStorage temp_storage[warps_per_thread_block];

    int warp_id       = threadIdx.x / 32;
    int sum_dct_block = warp_reduce(temp_storage[warp_id]).Sum(sum_tid);
    if (threadIdx.x % 32 == 0) {
        max_squared_diff_per_block[tid / warp_size] = sum_dct_block;
    }
}

} // namespace

bool psnr(
    std::vector<float>& vals,
    const std::vector<gpu_buf<uint8_t>>& pixels_a,
    const std::vector<gpu_buf<uint8_t>>& pixels_b,
    const std::vector<int>& num_blocks)
{
    assert(vals.size() == pixels_a.size());
    assert(pixels_a.size() == pixels_b.size());
    assert(pixels_b.size() == num_blocks.size());
    const int num_components = vals.size();

    // reduces 64 values per segment. max diff is 255, max squared diff is 65025.
    //   max accumulated value is 4161600, which fits in 22 bits.
    gpu_buf<int> d_max_squared_diff_per_block;

    gpu_buf<int> d_max_squared_diff;

    gpu_buf<char> d_tmp;
    for (int c = 0; c < num_components; ++c) {
        const int num_blocks_c = num_blocks[c];
        RETURN_IF_ERR_CUDA(d_max_squared_diff_per_block.resize(num_blocks_c));

        const int num_elements_per_kernel_block = thread_block_size * elements_per_thread;
        const int num_kernel_blocks =
            (num_blocks_c * dct_block_size + num_elements_per_kernel_block - 1) /
            num_elements_per_kernel_block;

        calc_max_squared_diff<<<num_kernel_blocks, thread_block_size, 0, nullptr>>>(
            d_max_squared_diff_per_block.ptr, pixels_a[c].ptr, pixels_b[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());

        RETURN_IF_ERR_CUDA(d_max_squared_diff.resize(1));

        size_t num_tmp = 0;
        RETURN_IF_ERR_CUDA(cub::DeviceReduce::Max(
            nullptr,
            num_tmp,
            d_max_squared_diff_per_block.ptr,
            d_max_squared_diff.ptr,
            num_blocks_c));
        RETURN_IF_ERR_CUDA(d_tmp.resize(num_tmp));
        RETURN_IF_ERR_CUDA(cub::DeviceReduce::Max(
            d_tmp.ptr,
            num_tmp,
            d_max_squared_diff_per_block.ptr,
            d_max_squared_diff.ptr,
            num_blocks_c));

        int h_max_squared_diff = 0;
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            &h_max_squared_diff, d_max_squared_diff.ptr, sizeof(int), cudaMemcpyDeviceToHost));

        vals[c] = 20.f * log10f(255.f) - 10.f * log10f((1.f / dct_block_size) * h_max_squared_diff);
    }

    return true;
}
