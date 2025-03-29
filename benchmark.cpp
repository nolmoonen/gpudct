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
#include <curand.h>
#include <nvml.h>

#include <functional>
#include <limits>
#include <numeric>
#include <stdint.h>
#include <vector>

namespace {

/// \brief Able to lock GPU clocks and unlock in destructor.
struct gpu_clock_lock {
    gpu_clock_lock() : locked(false) {}

    gpu_clock_lock(const gpu_clock_lock&)            = delete;
    gpu_clock_lock(gpu_clock_lock&&)                 = default;
    gpu_clock_lock& operator=(const gpu_clock_lock&) = delete;
    gpu_clock_lock& operator=(gpu_clock_lock&&)      = default;

    ~gpu_clock_lock()
    {
        if (locked) {
            nvmlDeviceResetGpuLockedClocks(nvml_device);
        }
    }

    bool lock(int device_id)
    {
        // see https://github.com/NVIDIA/nvbench/blob/1efed5f8e13903a12d9348dab2f7ff0fe6b8ecfd/nvbench/device_info.cu

        constexpr int pci_id_len = 13;
        char pci_id[pci_id_len];
        RETURN_IF_ERR_CUDA(cudaDeviceGetPCIBusId(pci_id, pci_id_len, device_id));

        RETURN_IF_ERR_NVML(nvmlInit());

        RETURN_IF_ERR_NVML(nvmlDeviceGetHandleByPciBusId(pci_id, &nvml_device));
        RETURN_IF_ERR_NVML(nvmlDeviceSetPersistenceMode(nvml_device, NVML_FEATURE_ENABLED));

        // set clocks to base thermal design power
        RETURN_IF_ERR_NVML(nvmlDeviceSetGpuLockedClocks(
            nvml_device,
            static_cast<unsigned int>(NVML_CLOCK_LIMIT_ID_TDP),
            static_cast<unsigned int>(NVML_CLOCK_LIMIT_ID_TDP)));

        // unfortunately nvmlDeviceSetMemoryLockedClocks is not supported for Turing

        locked = true;

        return true;
    }

    nvmlDevice_t nvml_device;
    bool locked;
};

bool benchmark(const char* filename)
{
    cudaStream_t stream;
    RETURN_IF_ERR_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start;
    RETURN_IF_ERR_CUDA(cudaEventCreate(&start));

    cudaEvent_t stop;
    RETURN_IF_ERR_CUDA(cudaEventCreate(&stop));

    std::vector<gpu_buf<uint8_t>> d_pixels;
    std::vector<gpu_buf<int16_t>> d_coeffs;
    std::vector<gpu_buf<uint16_t>> d_qtables;
    std::vector<int> num_blocks_aligned;
    int num_iterations;

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
    assert(num_blocks_lcm <= 64); // assume some reasonable rounding factor

    if (filename == nullptr) {
        d_pixels.resize(1);
        d_coeffs.resize(1);
        d_qtables.resize(1);
        num_blocks_aligned.resize(1);
        num_iterations = 50;

        const size_t num_gib              = 2;
        const size_t num_bytes            = num_gib << 30;
        const size_t num_coeffs           = num_bytes / sizeof(int16_t);
        const unsigned int num_coeffs_lcm = dct_block_size * num_blocks_lcm;
        const size_t num_coeffs_aligned = num_coeffs_lcm * ceiling_div(num_coeffs, num_coeffs_lcm);

        static_assert(num_coeffs % dct_block_size == 0);
        num_blocks_aligned[0] = num_coeffs_aligned / dct_block_size;

        RETURN_IF_ERR(d_pixels[0].resize(num_coeffs_aligned));

        curandGenerator_t gen;
        RETURN_IF_ERR_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        RETURN_IF_ERR(d_coeffs[0].resize(num_coeffs_aligned));
        assert(num_coeffs_aligned % 2 == 0);
        RETURN_IF_ERR_CURAND(curandGenerate(
            gen, reinterpret_cast<uint32_t*>(d_coeffs[0].ptr), num_coeffs_aligned / 2));
        RETURN_IF_ERR_CURAND(curandDestroyGenerator(gen));

        std::vector<uint16_t> h_qtables(dct_block_size);
        std::iota(h_qtables.begin(), h_qtables.end(), 1);

        RETURN_IF_ERR(d_qtables[0].resize(dct_block_size));
        RETURN_IF_ERR_CUDA(cudaMemcpy(
            h_qtables.data(),
            d_qtables[0].ptr,
            dct_block_size * sizeof(uint16_t),
            cudaMemcpyHostToDevice));

        printf("finished generating random coefficients\n");
    } else {
        image img;
        RETURN_IF_ERR(load_coeffs(img, filename));
        const int num_components = img.coeffs.size();

        d_pixels.resize(num_components);
        d_coeffs.resize(num_components);
        d_qtables.resize(num_components);
        num_blocks_aligned.resize(num_components);
        num_iterations = 100;

        for (int c = 0; c < num_components; ++c) {
            const int num_blocks_c = img.num_blocks[c];
            const int num_blocks_c_aligned =
                ceiling_div(num_blocks_c, static_cast<unsigned int>(num_blocks_lcm)) *
                num_blocks_lcm;
            num_blocks_aligned[c] = num_blocks_c_aligned;

            const size_t num_elements = dct_block_size * num_blocks_c;
            assert(num_elements == img.coeffs[c].size());
            const size_t num_elements_aligned = dct_block_size * num_blocks_c_aligned;

            RETURN_IF_ERR(d_pixels[c].resize(num_elements_aligned));
            RETURN_IF_ERR(d_coeffs[c].resize(num_elements_aligned));
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
            RETURN_IF_ERR(d_qtables[c].resize(dct_block_size));
            RETURN_IF_ERR_CUDA(cudaMemcpy(
                d_qtables[c].ptr,
                img.qtable[c].data(),
                dct_block_size * sizeof(uint16_t),
                cudaMemcpyHostToDevice));
        }

        printf("finished loading image coefficients\n");
    }

    int device_id;
    RETURN_IF_ERR_CUDA(cudaGetDevice(&device_id));

    gpu_clock_lock lock;
    lock.lock(device_id);

    int memory_bus_bit_width;
    RETURN_IF_ERR_CUDA(
        cudaDeviceGetAttribute(&memory_bus_bit_width, cudaDevAttrGlobalMemoryBusWidth, device_id));

    int memory_khz;
    RETURN_IF_ERR_CUDA(cudaDeviceGetAttribute(&memory_khz, cudaDevAttrMemoryClockRate, device_id));

    const double read_bytes_per_second = (memory_bus_bit_width / 8.) * (memory_khz * size_t{1000});

    const double gib = 1024. * 1024. * 1024.;

    const double read_gib_per_second = read_bytes_per_second / gib;
    const double read_gb_per_second  = read_bytes_per_second / (1000. * 1000. * 1000.);
    printf("memory read bandwidth: %f GiB/s %f GB/s\n", read_gib_per_second, read_gb_per_second);

    int ecc_enabled;
    RETURN_IF_ERR_CUDA(cudaDeviceGetAttribute(&ecc_enabled, cudaDevAttrEccEnabled, device_id));
    if (ecc_enabled) {
        // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#theoretical-bandwidth-calculation
        printf("ecc is enabled, memory bandwidth is affected\n");
    }

    auto measure = [&](const char* name, std::function<bool()> dispatch) -> bool {
        float min  = std::numeric_limits<float>::max();
        float max  = std::numeric_limits<float>::lowest();
        double sum = 0.f;

        for (int i = 0; i < num_iterations; ++i) {
            RETURN_IF_ERR_CUDA(cudaEventRecord(start, stream));
            RETURN_IF_ERR(dispatch());
            RETURN_IF_ERR_CUDA(cudaEventRecord(stop, stream));
            RETURN_IF_ERR_CUDA(cudaEventSynchronize(stop));
            float ms;
            RETURN_IF_ERR_CUDA(cudaEventElapsedTime(&ms, start, stop));
            min = std::min(min, ms);
            max = std::max(max, ms);
            sum += ms;
        }

        size_t num_read_bytes_iter = 0;
        for (size_t c = 0; c < d_coeffs.size(); ++c) {
            num_read_bytes_iter += d_coeffs[c].num * sizeof(int16_t);
        }
        const size_t num_read_bytes = num_iterations * num_read_bytes_iter;

        const double throughput_read_avg = num_read_bytes / (sum / 1000.) / gib;
        const double throughput_read_max = num_read_bytes_iter / (min / 1000.) / gib;
        const double throughput_read_min = num_read_bytes_iter / (max / 1000.) / gib;
        const float ms_avg               = sum / num_iterations;

        printf(
            "%12s: %6.2f [%6.2f, %6.2f] GiB/s %10.5fms\n",
            name,
            throughput_read_avg,
            throughput_read_min,
            throughput_read_max,
            ms_avg);
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
    measure("no_shared", [&]() {
        RETURN_IF_ERR(idct_no_shared(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream));
        return true;
    });
    measure("next", [&]() {
        RETURN_IF_ERR(idct_next(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream));
        return true;
    });
    measure("next16", [&]() {
        RETURN_IF_ERR(idct_next16(d_pixels, d_coeffs, d_qtables, num_blocks_aligned, stream));
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
    if (argc == 1) {
        printf("synthetic benchmark mode.. ");
        return benchmark(nullptr) ? EXIT_SUCCESS : EXIT_FAILURE;
    } else if (argc == 2) {
        printf("image benchmark mode.. ");
        return benchmark(argv[1]) ? EXIT_SUCCESS : EXIT_FAILURE;
    } else {
        printf("usage: gpudct_benchmark <optional: jpeg file>\n");
        return EXIT_FAILURE;
    }
}
