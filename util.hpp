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

#ifndef GPUDCT_UTIL_HPP_
#define GPUDCT_UTIL_HPP_

#include <cuda_runtime.h>
#include <curand.h>
#include <nvml.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <type_traits>
#include <vector>

#include <setjmp.h>
#include <stdint.h>

#define RETURN_IF_ERR(call)                                                    \
    do {                                                                       \
        const bool _res = call;                                                \
        if (!_res) {                                                           \
            printf("%s returned false at " __FILE__ ":%d\n", #call, __LINE__); \
            return false;                                                      \
        }                                                                      \
    } while (false)

#define RETURN_IF_ERR_CUDA(call)                           \
    do {                                                   \
        const cudaError_t _err = call;                     \
        if (_err != cudaSuccess) {                         \
            printf(                                        \
                "%s returned \"%s\" at " __FILE__ ":%d\n", \
                #call,                                     \
                cudaGetErrorString(_err),                  \
                __LINE__);                                 \
            return false;                                  \
        }                                                  \
    } while (false)

#define RETURN_IF_ERR_CURAND(call)                                                               \
    do {                                                                                         \
        const curandStatus_t _err = call;                                                        \
        if (_err != CURAND_STATUS_SUCCESS) {                                                     \
            printf(                                                                              \
                "%s returned %d at " __FILE__ ":%d\n", #call, static_cast<int>(_err), __LINE__); \
            return false;                                                                        \
        }                                                                                        \
    } while (false)

#define RETURN_IF_ERR_NVML(call)                           \
    do {                                                   \
        const nvmlReturn_t _ret = call;                    \
        if (_ret != NVML_SUCCESS) {                        \
            printf(                                        \
                "%s returned \"%s\" at " __FILE__ ":%d\n", \
                #call,                                     \
                nvmlErrorString(_ret),                     \
                __LINE__);                                 \
            return false;                                  \
        }                                                  \
    } while (false)

constexpr int dct_block_size = 64;
constexpr int dct_block_dim  = 8;

// growing buffer, freed by destructor
template <typename T>
struct vector {
    vector() noexcept : ptr(nullptr), num(0) {}

    // https://stackoverflow.com/questions/8001823/how-to-enforce-move-semantics-when-a-vector-grows
    vector(const vector&)                = delete;
    vector(vector&&) noexcept            = default;
    vector& operator=(const vector&)     = delete;
    vector& operator=(vector&&) noexcept = default;

    [[nodiscard]] virtual bool resize(size_t num_elements) noexcept = 0;

    T* ptr;
    size_t num;
};

template <typename T>
struct gpu_buf : vector<T> {
    gpu_buf() noexcept : vector<T>() {}

    gpu_buf(const gpu_buf&)                = delete;
    gpu_buf(gpu_buf&&) noexcept            = default;
    gpu_buf& operator=(const gpu_buf&)     = delete;
    gpu_buf& operator=(gpu_buf&&) noexcept = default;

    ~gpu_buf() noexcept
    {
        if (this->ptr != nullptr) {
            cudaFree(this->ptr);
            this->ptr = nullptr;
            this->num = 0;
        }
    }

    [[nodiscard]] bool resize(size_t num_elements) noexcept
    {
        if (num_elements <= this->num) {
            return true;
        }

        if (this->ptr != nullptr) {
            RETURN_IF_ERR_CUDA(cudaFree(this->ptr));
            this->ptr = nullptr;
            this->num = 0;
        }

        RETURN_IF_ERR_CUDA(cudaMalloc(&(this->ptr), num_elements * sizeof(T)));
        this->num = num_elements;

        return true;
    }
};

template <typename T>
struct cpu_buf : vector<T> {
    cpu_buf() noexcept : vector<T>() {}

    cpu_buf(const cpu_buf&)                = delete;
    cpu_buf(cpu_buf&&) noexcept            = default;
    cpu_buf& operator=(const cpu_buf&)     = delete;
    cpu_buf& operator=(cpu_buf&&) noexcept = default;

    ~cpu_buf() noexcept
    {
        if (this->ptr != nullptr) {
            std::free(this->ptr);
            this->ptr = nullptr;
            this->num = 0;
        }
    }

    [[nodiscard]] bool resize(size_t num_elements) noexcept
    {
        if (num_elements <= this->num) {
            return true;
        }

        if (this->ptr != nullptr) {
            std::free(this->ptr);
            this->ptr = nullptr;
            this->num = 0;
        }

        T* pointer = static_cast<T*>(std::malloc(num_elements * sizeof(T)));
        if (pointer == nullptr) {
            return false;
        }
        this->ptr = pointer;
        this->num = num_elements;

        return true;
    }
};

template <
    typename T,
    typename U,
    std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<U>::value, int> = 0>
__device__ __host__ constexpr auto ceiling_div(const T a, const U b)
{
    return a / b + (a % b > 0 ? 1 : 0);
}

struct image {
    std::vector<std::vector<int16_t>> coeffs;
    std::vector<std::vector<uint16_t>> qtable;
    std::vector<int> num_blocks_x;
    std::vector<int> num_blocks_y;
    std::vector<int> num_blocks;
};

bool load_coeffs(image& img, const char* filename);

inline void write_ppm(
    const image& img, const std::string& filename, const std::vector<cpu_buf<uint8_t>>& pixels)
{
    assert(img.coeffs.size() == img.qtable.size());
    assert(img.qtable.size() == img.num_blocks_x.size());
    assert(img.num_blocks_x.size() == img.num_blocks_y.size());
    assert(img.num_blocks_y.size() == img.num_blocks.size());
    const int num_components = img.coeffs.size();

    if (num_components != 3 || img.num_blocks_x[0] != img.num_blocks_x[1] ||
        img.num_blocks_x[1] != img.num_blocks_x[2] || img.num_blocks_y[0] != img.num_blocks_y[1] ||
        img.num_blocks_y[1] != img.num_blocks_y[2]) {
        printf("simple ppm writer cannot process\n");
        return;
    }

    const int num_blocks_x = img.num_blocks_x[0];
    const int num_blocks_y = img.num_blocks_y[0];

    const int size_x = 8 * num_blocks_x;
    const int size_y = 8 * num_blocks_y;

    // just output the full blocks
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file << "P6\n" << size_x << " " << size_y << "\n255\n";
    for (int y = 0; y < size_y; ++y) {
        const int by = y / 8;
        const int yy = y % 8;
        for (int x = 0; x < size_x; ++x) {
            const int bx = x / 8;
            const int xx = x % 8;

            const size_t i = dct_block_size * (num_blocks_x * by + bx) + 8 * yy + xx;

            const uint8_t cy = pixels[0].ptr[i];
            const uint8_t cb = pixels[1].ptr[i];
            const uint8_t cr = pixels[2].ptr[i];

            auto cvt = [](float a) { return std::clamp(std::roundf(a), 0.f, 255.f); };

            // https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
            const uint8_t r = cvt(cy + 1.402f * (cr - 128.f));
            const uint8_t g = cvt(cy - .344136f * (cb - 128.f) - .714136f * (cr - 128.f));
            const uint8_t b = cvt(cy + 1.772f * (cb - 128.f));

            file << r << g << b;
        }
    }
}

#endif // GPUDCT_UTIL_HPP_
