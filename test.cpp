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
#include <stdint.h>
#include <vector>

namespace {

bool pixels_dtoh(
    std::vector<std::vector<uint8_t>>& h_pixels, const std::vector<gpu_buf<uint8_t>>& d_pixels)
{
    assert(h_pixels.size() == d_pixels.size());
    const int num_components = h_pixels.size();
    for (int c = 0; c < num_components; ++c) {
        // TODO d_pixels[c].n may be bigger. keep track of current size as well?
        assert(h_pixels[c].size() <= static_cast<size_t>(d_pixels[c].num));
        const int num_elements = h_pixels[c].size();
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            h_pixels[c].data(),
            d_pixels[c].ptr,
            num_elements * sizeof(uint8_t),
            cudaMemcpyDeviceToHost));
    }

    return true;
}

bool pixels_htod(
    std::vector<gpu_buf<uint8_t>>& d_pixels, const std::vector<std::vector<uint8_t>>& h_pixels)
{
    assert(d_pixels.size() == h_pixels.size());
    const int num_components = d_pixels.size();
    for (int c = 0; c < num_components; ++c) {
        // TODO d_pixels[c].n may be bigger. keep track of current size as well?
        assert(static_cast<size_t>(d_pixels[c].num) >= h_pixels[c].size());
        const int num_elements = d_pixels[c].num;
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

bool test(const char* filename)
{
    image img;
    RETURN_IF_ERR(load_coeffs(img, filename));
    assert(img.coeffs.size() == img.qtable.size());
    assert(img.qtable.size() == img.num_blocks_x.size());
    assert(img.num_blocks_x.size() == img.num_blocks_y.size());
    assert(img.num_blocks_y.size() == img.num_blocks.size());
    const int num_components = img.coeffs.size();

    std::vector<gpu_buf<uint8_t>> d_pixels_gold(num_components);
    std::vector<gpu_buf<uint8_t>> d_pixels(num_components);

    std::vector<std::vector<uint8_t>> h_pixels(num_components);
    std::vector<gpu_buf<int16_t>> d_coeffs(num_components);
    std::vector<gpu_buf<uint16_t>> d_qtables(num_components);
    std::vector<float> vals(num_components);

    for (int c = 0; c < num_components; ++c) {
        const size_t num_elements = dct_block_size * img.num_blocks[c];
        assert(num_elements == img.coeffs[c].size());

        h_pixels[c].resize(num_elements);
        RETURN_IF_ERR_CUDA(d_pixels[c].resize(num_elements));
        RETURN_IF_ERR_CUDA(d_coeffs[c].resize(num_elements));
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            d_coeffs[c].ptr,
            img.coeffs[c].data(),
            num_elements * sizeof(int16_t),
            cudaMemcpyHostToDevice));
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
    RETURN_IF_ERR(pixels_htod(d_pixels_gold, h_pixels));

    cudaStream_t stream = nullptr;

    // gpu implementation
    idct(d_pixels, d_coeffs, d_qtables, img.num_blocks, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("naive", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels));
        const std::string filename_out(std::string(filename) + "_naive.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    idct_lut(d_pixels, d_coeffs, d_qtables, img.num_blocks, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("lut", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels));
        const std::string filename_out(std::string(filename) + "_lut.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    idct_seperable(d_pixels, d_coeffs, d_qtables, img.num_blocks, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("seperable", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels));
        const std::string filename_out(std::string(filename) + "_seperable.ppm");
        write_ppm(img, filename_out, h_pixels);
    }

    idct_gpujpeg(d_pixels, d_coeffs, d_qtables, img.num_blocks, stream);
    RETURN_IF_ERR(psnr(vals, d_pixels_gold, d_pixels, img.num_blocks));
    print_result("gpujpeg", vals);
    if (true) {
        RETURN_IF_ERR(pixels_dtoh(h_pixels, d_pixels));
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
