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

#include <dct.hpp>
#include <gpujpeg.hpp>

#include <array>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <stdint.h>
#include <vector>

namespace {

bool pixels_dtoh(
    std::vector<std::vector<uint8_t>>& h_pixels,
    const std::vector<gpu_buf<uint8_t>>& d_pixels,
    const std::vector<int>& num_blocks)
{
    assert(h_pixels.size() == d_pixels.size());
    assert(d_pixels.size() == num_blocks.size());
    const int num_components = h_pixels.size();
    for (int c = 0; c < num_components; ++c) {
        const int num_elements = dct_block_size * num_blocks[c];
        assert(h_pixels[c].size() >= static_cast<size_t>(num_elements));
        assert(d_pixels[c].num >= num_elements);
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            h_pixels[c].data(),
            d_pixels[c].ptr,
            num_elements * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));
    }

    return true;
}

bool pixels_htod(
    std::vector<gpu_buf<uint8_t>>& d_pixels,
    const std::vector<std::vector<uint8_t>>& h_pixels,
    const std::vector<int>& num_blocks)
{
    assert(d_pixels.size() == h_pixels.size());
    assert(h_pixels.size() == num_blocks.size());
    const int num_components = d_pixels.size();
    for (int c = 0; c < num_components; ++c) {
        const int num_elements = dct_block_size * num_blocks[c];
        assert(d_pixels[c].num >= num_elements);
        assert(h_pixels[c].size() >= static_cast<size_t>(num_elements));
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            d_pixels[c].ptr,
            h_pixels[c].data(),
            num_elements * sizeof(uint8_t),
            cudaMemcpyHostToDevice));
    }

    return true;
}

void print_result(const char* name, const std::vector<float>& vals)
{
    printf("%12s [", name);
    for (size_t i = 0; i < vals.size(); ++i) {
        printf("%f%s", vals[i], i + 1 < vals.size() ? ", " : "");
    }
    printf("]\n");
}

bool clear_pixels(std::vector<gpu_buf<uint8_t>>& d_pixels)
{
    for (size_t c = 0; c < d_pixels.size(); ++c) {
        RETURN_IF_ERR_CUDA(cudaMemset(d_pixels[c].ptr, 0, d_pixels[c].num * sizeof(uint8_t)));
    }

    return true;
}

bool test(const char* filename)
{
    image img;
    RETURN_IF_ERR(load_coeffs(img, filename));
    assert(img.coeffs.size() == img.qtable.size());
    assert(img.qtable.size() == img.num_blocks_x.size());
    assert(img.num_blocks_x.size() == img.num_blocks_y.size());
    assert(img.num_blocks_y.size() == img.num_blocks.size());
    const int num_components = img.coeffs.size();

    std::vector<int> num_blocks_aligned(num_components);

    std::vector<gpu_buf<uint8_t>> d_pixels_gold(num_components);
    std::vector<gpu_buf<uint8_t>> d_pixels(num_components);

    std::vector<std::vector<uint8_t>> h_pixels(num_components);
    std::vector<gpu_buf<int16_t>> d_coeffs(num_components);
    std::vector<gpu_buf<uint16_t>> d_qtables(num_components);
    std::vector<float> vals(num_components);

    const int num_blocks_lcm = std::lcm(
        std::lcm(
            std::lcm(
                get_num_idct_blocks_per_thread_block_next16(),
                get_num_idct_blocks_per_thread_block_naive()),
            std::lcm(
                get_num_idct_blocks_per_thread_block_lut(),
                get_num_idct_blocks_per_thread_block_next())),
        std::lcm(
            std::lcm(
                get_num_idct_blocks_per_thread_block_seperable(),
                get_num_idct_blocks_per_thread_block_no_shared()),
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

        h_pixels[c].resize(num_elements_aligned);
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
        RETURN_IF_ERR_CUDA(d_pixels_gold[c].resize(num_elements));

        assert(img.qtable[c].size() == dct_block_size);
        RETURN_IF_ERR_CUDA(d_qtables[c].resize(dct_block_size));
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            d_qtables[c].ptr,
            img.qtable[c].data(),
            dct_block_size * sizeof(uint16_t),
            cudaMemcpyHostToDevice));
    }

    // gold implementation
    idct_cpu(h_pixels, img.coeffs, img.qtable, img.num_blocks);
    if (true) {
        const std::string filename_out(std::string(filename) + "_cpu.ppm");
        write_ppm(img, filename_out, h_pixels);
    }
    RETURN_IF_ERR(pixels_htod(d_pixels_gold, h_pixels, img.num_blocks));

    cudaStream_t stream = nullptr;

    // gpu implementations
    idct_naive(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("naive", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels, img.num_blocks));
        const std::string filename_out(std::string(filename) + "_naive.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    RETURN_IF_ERR(clear_pixels(d_pixels));

    idct_lut(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("lut", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels, img.num_blocks));
        const std::string filename_out(std::string(filename) + "_lut.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    RETURN_IF_ERR(clear_pixels(d_pixels));

    idct_seperable(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("seperable", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels, img.num_blocks));
        const std::string filename_out(std::string(filename) + "_seperable.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    RETURN_IF_ERR(clear_pixels(d_pixels));

    idct_decomposed(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("decomposed", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels, img.num_blocks));
        const std::string filename_out(std::string(filename) + "_decomposed.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    RETURN_IF_ERR(clear_pixels(d_pixels));

    idct_no_shared(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("no shared", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels, img.num_blocks));
        const std::string filename_out(std::string(filename) + "_no_shared.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    RETURN_IF_ERR(clear_pixels(d_pixels));

    idct_next(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("next", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels, img.num_blocks));
        const std::string filename_out(std::string(filename) + "_next.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    RETURN_IF_ERR(clear_pixels(d_pixels));

    idct_next16(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("next16", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels, img.num_blocks));
        const std::string filename_out(std::string(filename) + "_next16.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    RETURN_IF_ERR(clear_pixels(d_pixels));

    idct_gpujpeg(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("gpujpeg", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels, img.num_blocks));
        const std::string filename_out(std::string(filename) + "_gpujpeg.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        printf("usage: gpudct_test <jpeg file>\n");
        return EXIT_FAILURE;
    }

    if (!test(argv[1])) {
        return EXIT_FAILURE;
    }
}
