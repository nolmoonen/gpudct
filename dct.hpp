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

#ifndef GPUDCT_DCT_HPP_
#define GPUDCT_DCT_HPP_

#include "util.hpp"

#include <cuda_runtime.h>

#include <stdint.h>
#include <vector>

void idct_cpu(
    std::vector<cpu_buf<uint8_t>>& pixels,
    const std::vector<std::vector<int16_t>>& coeffs,
    const std::vector<std::vector<uint16_t>>& qtable,
    const std::vector<int>& num_blocks);

// needed for rounding up the allocation
int get_num_idct_blocks_per_thread_block_naive();

bool idct_naive(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

// needed for rounding up the allocation
int get_num_idct_blocks_per_thread_block_lut();

bool idct_lut(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

// needed for rounding up the allocation
int get_num_idct_blocks_per_thread_block_seperable();

bool idct_seperable(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

// needed for rounding up the allocation
int get_num_idct_blocks_per_thread_block_decomposed();

bool idct_decomposed(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

// needed for rounding up the allocation
int get_num_idct_blocks_per_thread_block_no_shared();

bool idct_no_shared(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

// needed for rounding up the allocation
int get_num_idct_blocks_per_thread_block_next();

bool idct_next(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

// needed for rounding up the allocation
int get_num_idct_blocks_per_thread_block_next16();

bool idct_next16(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

#endif // GPUDCT_DCT_HPP_
