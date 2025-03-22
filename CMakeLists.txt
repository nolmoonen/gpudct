# Copyright (c) 2025 Nol Moonen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# 3.17 FindCUDAToolkit
# 3.24 CUDA_ARCHITECTURES native
cmake_minimum_required(VERSION 3.17)

project(gpudct CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

# not always best practice, however for this project the target is fixed 
set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# version 80 required for jpeg_mem_src, jpeg_read_coefficients
find_package(JPEG REQUIRED 80)

add_library(gpudct STATIC dct.cu util.cpp gpujpeg.cu)
target_link_libraries(gpudct PRIVATE JPEG::JPEG CUDA::cudart_static)
target_include_directories(gpudct PUBLIC .)
# target_compile_options(gpudct PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--fast-math>)

add_executable(gpudct_test test.cpp psnr.cu)
target_link_libraries(gpudct_test PRIVATE gpudct CUDA::cudart_static)

add_executable(gpudct_benchmark benchmark.cpp)
target_link_libraries(gpudct_benchmark PRIVATE gpudct CUDA::cudart_static)
