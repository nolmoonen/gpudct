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

// clang-format off
#include <cstdio>
#include <jpeglib.h>
// clang-format on

namespace {

struct my_error_mgr {
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

METHODDEF(void)
my_error_exit(j_common_ptr cinfo)
{
    my_error_mgr* myerr = reinterpret_cast<my_error_mgr*>(cinfo->err);

    (*cinfo->err->output_message)(cinfo);

    longjmp(myerr->setjmp_buffer, 1);
}

} // namespace

bool load_coeffs(image& img, const char* filename)
{
    // TODO persist jpeg state

    std::ifstream stream;
    stream.open(filename, std::ios_base::binary | std::ios_base::in);
    if (stream.fail()) {
        printf("failed to open %s\n", filename);
        return false;
    }
    stream.seekg(0, std::ios_base::end);
    const std::streampos size = stream.tellg();
    stream.seekg(0, std::ios_base::beg);
    std::vector<uint8_t> file_mem(size);
    stream.read(reinterpret_cast<char*>(file_mem.data()), file_mem.size());

    jpeg_decompress_struct cinfo;
    my_error_mgr jerr;

    cinfo.err           = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }

    jpeg_create_decompress(&cinfo);

    jpeg_mem_src(&cinfo, file_mem.data(), file_mem.size());

    const bool require_image = true;
    jpeg_read_header(&cinfo, require_image);

    // coefficients are in raster order
    jvirt_barray_ptr* h_coeff = jpeg_read_coefficients(&cinfo);

    for (int c = 0; c < cinfo.num_components; ++c) {
        const int by      = cinfo.comp_info[c].height_in_blocks;
        const int bx      = cinfo.comp_info[c].width_in_blocks;
        const int ssy     = cinfo.comp_info[c].v_samp_factor;
        const size_t size = by * bx * 64;

        std::vector<int16_t> h_coeff_buffer(size);

        for (int y = 0; y < by; y += ssy) {
            const size_t row_size = bx * 64;
            const size_t off      = y * row_size;
            const int num_rows    = std::min(by - y, ssy);
            // access multiple rows at a time, see jctrans.c for reference
            int16_t(**h_coeffc)[64] = cinfo.mem->access_virt_barray(
                reinterpret_cast<j_common_ptr>(&cinfo), h_coeff[c], y, num_rows, false);
            std::memcpy(
                h_coeff_buffer.data() + off, **h_coeffc, row_size * num_rows * sizeof(int16_t));
        }

        img.coeffs.push_back(std::move(h_coeff_buffer));

        std::vector<uint16_t> qtable(64);
        if (cinfo.comp_info[c].quant_table == nullptr) {
            printf("no quantization table available\n");
            return false;
        }

        std::memcpy(qtable.data(), cinfo.comp_info[c].quant_table, 64 * sizeof(uint16_t));
        img.qtable.push_back(std::move(qtable));

        img.num_blocks_x.push_back(bx);
        img.num_blocks_y.push_back(by);
        img.num_blocks.push_back(bx * by);
    }

    jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);

    return true;
}
