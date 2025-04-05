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
#include <thrust/iterator/zip_iterator.h>

#include <cuda_runtime.h>

#include <cmath>
#include <stdint.h>

namespace {

__global__ void calc_squared_diff(
    uint16_t* squared_diff, const uint8_t* pixels_a, const uint8_t* pixels_b, int num_elements)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= num_elements) {
        return;
    }

    const int diff    = int{pixels_a[tid]} - pixels_b[tid];
    squared_diff[tid] = diff * diff;
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

    gpu_buf<char> d_tmp;
    gpu_buf<uint16_t> d_squared_diff;
    gpu_buf<size_t> d_sum_squared_diff;
    for (int c = 0; c < num_components; ++c) {
        const int num_blocks_c   = num_blocks[c];
        const int num_elements_c = dct_block_size * num_blocks_c;

        RETURN_IF_ERR(d_squared_diff.resize(num_elements_c));

        const int kernel_block_size = 256;
        const int num_kernel_blocks = (num_elements_c + kernel_block_size - 1) / kernel_block_size;

        calc_squared_diff<<<num_kernel_blocks, kernel_block_size, 0, nullptr>>>(
            d_squared_diff.ptr, pixels_a[c].ptr, pixels_b[c].ptr, num_elements_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());

        RETURN_IF_ERR(d_sum_squared_diff.resize(1));

        size_t num_tmp = 0;
        RETURN_IF_ERR_CUDA(
            cub::DeviceReduce::Sum(
                nullptr, num_tmp, d_squared_diff.ptr, d_sum_squared_diff.ptr, num_elements_c));
        RETURN_IF_ERR(d_tmp.resize(num_tmp));
        RETURN_IF_ERR_CUDA(
            cub::DeviceReduce::Sum(
                d_tmp.ptr, num_tmp, d_squared_diff.ptr, d_sum_squared_diff.ptr, num_elements_c));

        size_t h_sum_squared_diff = 0;
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            &h_sum_squared_diff, d_sum_squared_diff.ptr, sizeof(size_t), cudaMemcpyDeviceToHost));

        const double mse = h_sum_squared_diff / static_cast<double>(num_elements_c);

        vals[c] = 20. * log10(255.) - 10. * log10(mse);
    }

    return true;
}
