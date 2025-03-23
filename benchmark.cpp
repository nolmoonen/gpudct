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

#include "util.hpp"

#include <dct.hpp>
#include <gpujpeg.hpp>

#include <cuda_runtime.h>

#include <functional>
#include <numeric>
#include <stdint.h>
#include <vector>

namespace {

bool benchmark(const char* filename)
{
    image img;
    RETURN_IF_ERR(load_coeffs(img, filename));
    const int num_components = img.coeffs.size();

    std::vector<int> num_blocks_aligned(num_components);

    std::vector<gpu_buf<uint8_t>> d_pixels(num_components);

    std::vector<gpu_buf<int16_t>> d_coeffs(num_components);
    std::vector<gpu_buf<uint16_t>> d_qtables(num_components);

    const int num_blocks_lcm = std::lcm(
        std::lcm(
            get_num_idct_blocks_per_thread_block_naive(),
            get_num_idct_blocks_per_thread_block_lut()),
        std::lcm(
            get_num_idct_blocks_per_thread_block_seperable(),
            std::lcm(
                get_num_idct_blocks_per_thread_block_decomposed(),
                get_num_idct_blocks_per_thread_block_gpujpeg())));

    for (int c = 0; c < num_components; ++c) {
        const int num_blocks_c = img.num_blocks[c];
        const int num_blocks_c_aligned =
            ceiling_div(num_blocks_c, static_cast<unsigned int>(num_blocks_lcm)) * num_blocks_lcm;
        num_blocks_aligned[c] = num_blocks_c_aligned;

        const size_t num_elements = dct_block_size * num_blocks_c;
        assert(num_elements == img.coeffs[c].size());
        const size_t num_elements_aligned = dct_block_size * num_blocks_c_aligned;

        RETURN_IF_ERR_CUDA(d_pixels[c].resize(num_elements_aligned));
        RETURN_IF_ERR_CUDA(d_coeffs[c].resize(num_elements_aligned));
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            d_coeffs[c].ptr,
            img.coeffs[c].data(),
            num_elements * sizeof(int16_t),
            cudaMemcpyHostToDevice));
        RETURN_IF_ERR_CUDA(cudaMemset(
            d_coeffs[c].ptr + num_elements,
            0,
            (num_elements_aligned - num_elements) * sizeof(int16_t)));

        assert(img.qtable[c].size() == dct_block_size);
        RETURN_IF_ERR_CUDA(d_qtables[c].resize(dct_block_size));
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            d_qtables[c].ptr,
            img.qtable[c].data(),
            dct_block_size * sizeof(uint16_t),
            cudaMemcpyHostToDevice));
    }

    cudaStream_t stream;
    RETURN_IF_ERR_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start;
    RETURN_IF_ERR_CUDA(cudaEventCreate(&start));

    cudaEvent_t stop;
    RETURN_IF_ERR_CUDA(cudaEventCreate(&stop));

    auto measure =
        [&stream, &start, &stop](const char* name, std::function<bool()> dispatch) -> bool {
        RETURN_IF_ERR_CUDA(cudaEventRecord(start, stream));

        const int num_iters = 100;
        for (int i = 0; i < num_iters; ++i) {
            RETURN_IF_ERR(dispatch());
        }
        RETURN_IF_ERR_CUDA(cudaEventRecord(stop, stream));
        RETURN_IF_ERR_CUDA(cudaEventSynchronize(stop));

        float ms_sum = 0;
        RETURN_IF_ERR_CUDA(cudaEventElapsedTime(&ms_sum, start, stop));
        const float ms_avg = ms_sum / num_iters;
        printf("%12s: %10.5fms\n", name, ms_avg);
        return true;
    };

    measure("naive", [&]() {
        RETURN_IF_ERR(idct_naive(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream));
        return true;
    });
    measure("lut", [&]() {
        RETURN_IF_ERR(idct_lut(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream));
        return true;
    });
    measure("seperable", [&]() {
        RETURN_IF_ERR(idct_seperable(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream));
        return true;
    });
    measure("decomposed", [&]() {
        RETURN_IF_ERR(idct_decomposed(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream));
        return true;
    });
    measure("gpujpeg", [&]() {
        RETURN_IF_ERR(idct_gpujpeg(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream));
        return true;
    });

    RETURN_IF_ERR_CUDA(cudaEventDestroy(start));
    RETURN_IF_ERR_CUDA(cudaEventDestroy(stop));

    RETURN_IF_ERR_CUDA(cudaStreamDestroy(stream));

    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("usage: gpudct_benchmark <jpeg file>\n");
        return EXIT_FAILURE;
    }

    if (!benchmark(argv[1])) {
        return EXIT_FAILURE;
    }
}
