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
    std::vector<std::vector<uint8_t>>& pixels,
    const std::vector<std::vector<int16_t>>& coeffs,
    const std::vector<std::vector<uint16_t>>& qtable,
    const std::vector<int>& num_blocks);

bool idct(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

bool idct_lut(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

bool idct_seperable(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream);

#endif // GPUDCT_DCT_HPP_
