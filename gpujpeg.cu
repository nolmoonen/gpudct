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

/**
 * @file
 * Copyright (c) 2011-2020, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gpujpeg.hpp"
#include "util.hpp"

#include <stdint.h>
#include <vector>

/*
  * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
  *
  * Please refer to the NVIDIA end user license agreement (EULA) associated
  * with this source code for terms and conditions that govern your use of
  * this software. Any use, reproduction, disclosure, or distribution of
  * this software and related documentation outside the terms of the EULA
  * is strictly prohibited.
  *
  */

#define GPUJPEG_BLOCK_SIZE 8
#define GPUJPEG_BLOCK_SQUARED_SIZE 64
#define GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE (GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE * 8)

#define GPUJPEG_IDCT_BLOCK_X 8
#define GPUJPEG_IDCT_BLOCK_Y 8
#define GPUJPEG_IDCT_BLOCK_Z 2

//PTX code for IDCT (GPUJPEG_IDCT_GPU_KERNEL_INPLACE macro) should be a bit faster
//but maybe won't work for newer CCs
#define GPUJPEG_IDCT_USE_ASM 0

/** Fast integer multiplication */
#define FMUL(x, y) (__mul24(x, y))
//#define FMUL(x,y)   ((x)*(y))

// X block count which will be processed by one thread block
#define GPUJPEG_DCT_BLOCK_COUNT_X 4
// Y block count which will be processed by one thread block
#define GPUJPEG_DCT_BLOCK_COUNT_Y 4

// Thread block width
#define GPUJPEG_DCT_THREAD_BLOCK_WIDTH (GPUJPEG_BLOCK_SIZE * GPUJPEG_DCT_BLOCK_COUNT_X)
// Thread block height
#define GPUJPEG_DCT_THREAD_BLOCK_HEIGHT (GPUJPEG_BLOCK_SIZE * GPUJPEG_DCT_BLOCK_COUNT_Y)

// Stride of shared memory buffer (short kernel)
#define GPUJPEG_DCT_THREAD_BLOCK_STRIDE (GPUJPEG_DCT_THREAD_BLOCK_WIDTH + 4)

#define IMAD(a, b, c) (((a) * (b)) + (c))
#define IMUL(a, b) ((a) * (b))

#define SIN_1_4 0x5A82
#define COS_1_4 0x5A82
#define SIN_1_8 0x30FC
#define COS_1_8 0x7642

#define OSIN_1_16 0x063E
#define OSIN_3_16 0x11C7
#define OSIN_5_16 0x1A9B
#define OSIN_7_16 0x1F63

#define OCOS_1_16 0x1F63
#define OCOS_3_16 0x1A9B
#define OCOS_5_16 0x11C7
#define OCOS_7_16 0x063E

namespace {

/**
  * Package of 2 shorts into 1 int - designed to perform i/o by integers to avoid bank conflicts
  */
union PackedInteger {
    struct __align__(8) {
        int16_t hShort1;
        int16_t hShort2;
    };
    int32_t hInt;
};

/**
  * Converts fixed point value to short value
  */
__device__ inline int16_t unfixh(int x) { return (int16_t)((x + 0x8000) >> 16); }

/**
  * Converts fixed point value to short value
  */
__device__ inline int unfixo(int x) { return (x + 0x1000) >> 13; }

/** Constant memory copy of transposed quantization table pre-divided with DCT output weights. */
__constant__ float gpujpeg_dct_gpu_quantization_table_const[64];

/** Quantization table */
//TODO zmenit na float
__constant__ uint16_t gpujpeg_idct_gpu_quantization_table[64];

#if !GPUJPEG_IDCT_USE_ASM

/**
  * Performs in-place IDCT of vector of 8 elements (used to access rows 
  * or columns in a vector).
  * With a use of a scheme presented in Jie Liang - Approximating the DCT 
  * with the lifting scheme: systematic design and applications; online:
  * http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=910943
  *
  * @param V8 [IN/OUT] - Pointer to the first element of vector
  * @return None
  */
__device__ void gpujpeg_idct_gpu_kernel_inplace(float* V8)
{
    //costants which are used more than once
    const float koeficient[6] = {
        0.4142135623f, 0.3535533905f, 0.4619397662f, 0.1989123673f, 0.7071067811f, -2.0f};

    V8[2] *= 0.5411961f;
    V8[4] *= 0.509795579f;
    V8[5] *= 0.601344887f;

    V8[1] = (V8[0] - V8[1]) * koeficient[1];
    V8[0] = V8[0] * koeficient[4] - V8[1];

    V8[3] = V8[2] * koeficient[1] + V8[3] * koeficient[2];
    V8[2] = V8[3] * koeficient[0] - V8[2];

    V8[6] = V8[5] * koeficient[2] + V8[6] * koeficient[0];
    V8[5] = -0.6681786379f * V8[6] + V8[5];

    V8[7] = V8[4] * koeficient[3] + V8[7] * 0.49039264f;
    V8[4] = V8[7] * koeficient[3] - V8[4];

    //instead of float tmp = V8[1]; V8[1] = V8[2] + V8[1]; V8[2] = tmp - V8[2];
    //we use this two operations (with a use of a multiply-add instruction)
    V8[1] = V8[2] + V8[1];
    V8[2] = koeficient[5] * V8[2] + V8[1];

    V8[4] = V8[5] + V8[4];
    V8[5] = 2.0f * V8[5] - V8[4];

    V8[7] = V8[6] + V8[7];
    V8[6] = koeficient[5] * V8[6] + V8[7];

    V8[0] = V8[3] + V8[0];
    V8[3] = koeficient[5] * V8[3] + V8[0];

    V8[5] = V8[6] * koeficient[0] + V8[5];
    V8[6] = V8[5] * -koeficient[4] + V8[6];
    V8[5] = V8[6] * koeficient[0] + V8[5];

    V8[3] = V8[3] + V8[4];
    V8[4] = koeficient[5] * V8[4] + V8[3];

    V8[2] = V8[2] + V8[5];
    V8[5] = koeficient[5] * V8[5] + V8[2];

    V8[1] = V8[6] + V8[1];
    V8[6] = koeficient[5] * V8[6] + V8[1];

    V8[0] = V8[0] + V8[7];
    V8[7] = koeficient[5] * V8[7] + V8[0];
}
#else

#if __CUDA_ARCH__ >= 200
#define MULTIPLY_ADD "fma.rn.f32	"
#else
#define MULTIPLY_ADD "mad.f32	"
#endif

//instead of float tmp = V8[1]; V8[1] = V8[2] + V8[1]; V8[2] = tmp - V8[2];
//we use this two operations (with a use of a multiply-add instruction)
#define ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(x, y)                                       \
    "add.f32	" #x ", " #x ", " #y ";	\n\t" MULTIPLY_ADD #y ", " #y ", 0fc0000000, " #x ";	" \
    "\n\t"

/**
  * Performs in-place IDCT of 8 elements (rows or columns). A PTX implementation.
  * With a use of a scheme presented in Jie Liang - Approximating the DCT 
  * with the lifting scheme: systematic design and applications; online:
  * http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=910943
  */
#define GPUJPEG_IDCT_GPU_KERNEL_INPLACE(                                                                                          \
    in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3, out4, out5, out6, out7)                                       \
    asm(/* negreg register used for negating variables (e.g. for */ /* a * b - c we neagte c into negreg and use multiply-add) */ \
        "{.reg .f32 negreg;	\n\t"                                                                                                 \
                                                                                                                                  \
        "mul.f32	%9, %9, 0fbeb504f3;	\n\t" MULTIPLY_ADD                                                                \
        "%9, %8, 0f3eb504f3, %9;	\n\t"                                                                                            \
        "neg.f32 	negreg, %9;	\n\t" MULTIPLY_ADD "%8, %8, 0f3f3504f3, negreg;	\n\t"                                     \
                                                                                                                                  \
        "mul.f32	%10, %10, 0f3f0a8bd4;	\n\t"                                                                                      \
        "mul.f32	%11, %11, 0f3eec835e;	\n\t" MULTIPLY_ADD                                                                \
        "%11, %10, 0f3eb504f3, %11;	\n\t"                                                                                         \
        "neg.f32	%10, %10;	\n\t" MULTIPLY_ADD "%10, %11, 0f3ed413cd, %10;	\n\t"                                      \
                                                                                                                                  \
        "mul.f32	%13, %13, 0f3f19f1bd;	\n\t"                                                                                      \
        "mul.f32	%14, %14, 0f3ed4db31;	\n\t" MULTIPLY_ADD                                                                \
        "%14, %13, 0f3eec835e, %14;	\n\t" MULTIPLY_ADD "%13, %14, 0fbf2b0dc7, %13;	\n\t"                                      \
                                                                                                                                  \
        "mul.f32	%12, %12, 0f3f0281f7;	\n\t"                                                                                      \
        "mul.f32	%15, %15, 0f3efb14be;	\n\t" MULTIPLY_ADD                                                                \
        "%15, %12, 0f3e4bafaf, %15;	\n\t"                                                                                         \
        "neg.f32	%12, %12;	\n\t" MULTIPLY_ADD "%12, %15, 0f3e4bafaf, %12;	\n\t"                                      \
                                                                                                                                  \
        ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(% 9, % 10)                                                                     \
                                                                                                                                  \
            ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(% 12, % 13)                                                                \
                                                                                                                                  \
                ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(% 8, % 11)                                                             \
                                                                                                                                  \
                    ASM_X_PLUS_Y_SIMULTANEOUSLY_WITH_X_MINUS_Y(% 15, % 14)                                                        \
                                                                                                                                  \
                        MULTIPLY_ADD                                                                                              \
        "%13, %14, 0fbed413db, %13;	\n\t" MULTIPLY_ADD "%14, %13, 0f3f3504f3, %14;	\n\t"                                      \
        "neg.f32 negreg, %13;	\n\t" MULTIPLY_ADD "%13, %14, 0f3ed413cd, negreg;	\n\t"                                           \
                                                                                                                                  \
        /* writing into output registers */                                                                                       \
        "add.f32	%3, %11, %12;	\n\t"                                                                                              \
        "sub.f32	%4, %11, %12;	\n\t"                                                                                              \
                                                                                                                                  \
        "add.f32	%2, %10, %13;	\n\t"                                                                                              \
        "sub.f32	%5, %10, %13;	\n\t"                                                                                              \
                                                                                                                                  \
        "add.f32	%1, %14, %9;	\n\t"                                                                                               \
        "sub.f32	%6, %9, %14;	\n\t"                                                                                               \
                                                                                                                                  \
        "add.f32	%0, %8, %15;	\n\t"                                                                                               \
        "sub.f32	%7, %8, %15;	\n\t"                                                                                               \
        "}"                                                                                                                       \
                                                                                                                                  \
        : "=f"((out0)),                                                                                                           \
          "=f"((out1)),                                                                                                           \
          "=f"((out2)),                                                                                                           \
          "=f"((out3)),                                                                                                           \
          "=f"((out4)),                                                                                                           \
          "=f"((out5)),                                                                                                           \
          "=f"((out6)),                                                                                                           \
          "=f"((out7))                                                                                                            \
        : "f"((in0)),                                                                                                             \
          "f"((in1)),                                                                                                             \
          "f"((in2)),                                                                                                             \
          "f"((in3)),                                                                                                             \
          "f"((in4)),                                                                                                             \
          "f"((in5)),                                                                                                             \
          "f"((in6)),                                                                                                             \
          "f"((in7)));

#endif

/**
  * Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
  * image plane and outputs result to the array of coefficients. Float implementation.
  * This kernel is designed to process image by blocks of blocks8x8 that
  * utilize maximum warps capacity. Prepared for 8*8*2 threads in a block
  *
  * @param source             [IN]  - Source coefficients
  * @param output             [OUT] - Result coefficients
  * @param quantization_table [IN]  - Quantization table
  * @param num_blocks         [IN]  - Number of blocks
  * @return None
  */
__global__ void gpujpeg_idct_gpu_kernel(
    int16_t* source, uint8_t* result, uint16_t* quantization_table, int num_blocks)
{
    // TODO(nol) deal with `num_blocks`

    //here the grid is assumed to be only in x - it saves a few operations; if a larger
    //block count is used (e. g. GPUJPEG_IDCT_BLOCK_Z == 1), it would need to be adjusted,
    //the blockIdx.x not to exceed 65535. In the current state this function is good
    //enough for a 67.1 MPix picture (8K is 33.1 MPix)

    //the first block of picture processed in this thread block
    unsigned int picBlockNumber =
        (blockIdx.x) * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Z;

    //pointer to the begin of data for this thread block
    int16_t* sourcePtr = (int16_t*)(source) + picBlockNumber * 8;

    __shared__ float data[GPUJPEG_IDCT_BLOCK_Z][8][GPUJPEG_IDCT_BLOCK_Y][GPUJPEG_IDCT_BLOCK_X + 1];

    //variables to be used later more times (only one multiplication here)
    unsigned int z64 = threadIdx.z * 64;
    unsigned int x8  = threadIdx.x * 8;

    //data copying global -> shared, type casting int16_t -> float and dequantization.
    //16b reading gives only 50% efectivity but another ways are too complicated
    //so this proves to be the fastest way
#pragma unroll
    for (int i = 0; i < 8; i++) {
        data[threadIdx.z][i][threadIdx.x][threadIdx.y] =
            sourcePtr
                [x8 + threadIdx.y + i * GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y + z64 * 8] *
            quantization_table[threadIdx.x * 8 + threadIdx.y];
    }

    __syncthreads();

    float x[8];

    //kompilator delal hrozne psi kusy - zbytecne kopirovani konstant do
    //registru atp., bylo jednodussi napsat to v assembleru nez snazit se ho
    //presvedcit, aby nedelal blbosti; vsechny konstanty se pouzivaji primo
    //hodnotou, nestrkaji se zbytecne do registru

    //here the data are being processed by columns - each thread processes one column
#if GPUJPEG_IDCT_USE_ASM
    GPUJPEG_IDCT_GPU_KERNEL_INPLACE(
        data[threadIdx.z][threadIdx.x][0][threadIdx.y],
        data[threadIdx.z][threadIdx.x][4][threadIdx.y],
        data[threadIdx.z][threadIdx.x][6][threadIdx.y],
        data[threadIdx.z][threadIdx.x][2][threadIdx.y],
        data[threadIdx.z][threadIdx.x][7][threadIdx.y],
        data[threadIdx.z][threadIdx.x][5][threadIdx.y],
        data[threadIdx.z][threadIdx.x][3][threadIdx.y],
        data[threadIdx.z][threadIdx.x][1][threadIdx.y],

        data[threadIdx.z][threadIdx.x][0][threadIdx.y],
        data[threadIdx.z][threadIdx.x][1][threadIdx.y],
        data[threadIdx.z][threadIdx.x][2][threadIdx.y],
        data[threadIdx.z][threadIdx.x][3][threadIdx.y],
        data[threadIdx.z][threadIdx.x][4][threadIdx.y],
        data[threadIdx.z][threadIdx.x][5][threadIdx.y],
        data[threadIdx.z][threadIdx.x][6][threadIdx.y],
        data[threadIdx.z][threadIdx.x][7][threadIdx.y])
#else
    x[0] = data[threadIdx.z][threadIdx.x][0][threadIdx.y];
    x[1] = data[threadIdx.z][threadIdx.x][4][threadIdx.y];
    x[2] = data[threadIdx.z][threadIdx.x][6][threadIdx.y];
    x[3] = data[threadIdx.z][threadIdx.x][2][threadIdx.y];
    x[4] = data[threadIdx.z][threadIdx.x][7][threadIdx.y];
    x[5] = data[threadIdx.z][threadIdx.x][5][threadIdx.y];
    x[6] = data[threadIdx.z][threadIdx.x][3][threadIdx.y];
    x[7] = data[threadIdx.z][threadIdx.x][1][threadIdx.y];

    gpujpeg_idct_gpu_kernel_inplace(x);

    data[threadIdx.z][threadIdx.x][0][threadIdx.y] = x[0];
    data[threadIdx.z][threadIdx.x][1][threadIdx.y] = x[1];
    data[threadIdx.z][threadIdx.x][2][threadIdx.y] = x[2];
    data[threadIdx.z][threadIdx.x][3][threadIdx.y] = x[3];
    data[threadIdx.z][threadIdx.x][4][threadIdx.y] = x[4];
    data[threadIdx.z][threadIdx.x][5][threadIdx.y] = x[5];
    data[threadIdx.z][threadIdx.x][6][threadIdx.y] = x[6];
    data[threadIdx.z][threadIdx.x][7][threadIdx.y] = x[7];
#endif
    //between data writing and sync it's good to compute something useful
    // - the sync will be shorter.

    //output pointer (the begin for this thread block)
    unsigned int firstByteOfActualBlock = x8 + z64 + picBlockNumber;

    // NOTE(nol) specificied constant stride
    constexpr unsigned int output_stride = dct_block_dim;

    //output pointer for this thread + output row shift; each thread writes 1 row of an
    //output block (8B), threads [0 - 7] in threadIdx.x write blocks next to each other,
    //threads [1 - 7] in threadIdx.y write next rows of a block; threads [0 - 1] in
    //threadIdx.z write next 8 blocks
    uint8_t* resultPtr =
        ((uint8_t*)result) + firstByteOfActualBlock +
        (threadIdx.y + ((firstByteOfActualBlock / output_stride) * 7)) * output_stride;

    __syncthreads();

#if GPUJPEG_IDCT_USE_ASM
    //here the data are being processed by rows - each thread processes one row
    GPUJPEG_IDCT_GPU_KERNEL_INPLACE(
        data[threadIdx.z][threadIdx.x][threadIdx.y][0],
        data[threadIdx.z][threadIdx.x][threadIdx.y][4],
        data[threadIdx.z][threadIdx.x][threadIdx.y][6],
        data[threadIdx.z][threadIdx.x][threadIdx.y][2],
        data[threadIdx.z][threadIdx.x][threadIdx.y][7],
        data[threadIdx.z][threadIdx.x][threadIdx.y][5],
        data[threadIdx.z][threadIdx.x][threadIdx.y][3],
        data[threadIdx.z][threadIdx.x][threadIdx.y][1],

        x[0],
        x[1],
        x[2],
        x[3],
        x[4],
        x[5],
        x[6],
        x[7])
#else
    x[0] = data[threadIdx.z][threadIdx.x][threadIdx.y][0];
    x[1] = data[threadIdx.z][threadIdx.x][threadIdx.y][4];
    x[2] = data[threadIdx.z][threadIdx.x][threadIdx.y][6];
    x[3] = data[threadIdx.z][threadIdx.x][threadIdx.y][2];
    x[4] = data[threadIdx.z][threadIdx.x][threadIdx.y][7];
    x[5] = data[threadIdx.z][threadIdx.x][threadIdx.y][5];
    x[6] = data[threadIdx.z][threadIdx.x][threadIdx.y][3];
    x[7] = data[threadIdx.z][threadIdx.x][threadIdx.y][1];

    gpujpeg_idct_gpu_kernel_inplace(x);
#endif

    //output will be written by 8B (one row) which is the most effective way
    uint64_t tempResult;
    uint64_t* tempResultP = &tempResult;

#pragma unroll
    for (int i = 0; i < 8; i++) {
        //this would be faster but will work only for 100% quality otherwise some values overflow 255
        //((uint8_t*) tempResultP)[i] = __float2uint_rz(x[i] + ((float) 128.0));

        //cast float to uint8_t with saturation (.sat) which cuts values higher than
        //255 to 255 and smaller than 0 to 0; cuda can't use a reg smaller than 32b
        //(though it can convert to 8b for the saturation purposes and save to 32b reg)
        // uint32_t save;
        // asm("cvt.rni.u8.f32.sat	%0, %1;" : "=r"(save) : "f"(x[i] + ((float) 128.0)));

        // Following wokaround enables GPUJPEG with ZLUDA (see GH-90). May be slower
        // but not measurable because perhaps the computation time is masked by global
        // memory transfers.
        int save                   = rintf(x[i] + 128.0F);
        save                       = save < 0 ? 0 : save > 255 ? 255 : save;
        ((uint8_t*)tempResultP)[i] = save;
    }

    //writing result - one row of a picture block by a thread
    *((uint64_t*)resultPtr) = tempResult;
}

} // namespace

bool idct_gpujpeg(
    std::vector<gpu_buf<uint8_t>>& pixels,
    const std::vector<gpu_buf<int16_t>>& coeffs,
    const std::vector<gpu_buf<uint16_t>>& qtable,
    const std::vector<int>& num_blocks,
    cudaStream_t stream)
{
    assert(pixels.size() == coeffs.size());
    assert(coeffs.size() == qtable.size());
    assert(qtable.size() == num_blocks.size());
    const int num_components = pixels.size();

    for (int c = 0; c < num_components; ++c) {
        const int num_blocks_c = num_blocks[c];

        const unsigned int num_dct_blocks_per_thread_block =
            GPUJPEG_IDCT_BLOCK_X * GPUJPEG_IDCT_BLOCK_Y * GPUJPEG_IDCT_BLOCK_Z;

        // TODO(nol) prevent oob, either kernel check or rounding up allocation (better)
        assert(num_blocks_c % num_dct_blocks_per_thread_block = 0);

        const unsigned int num_elements_per_block =
            num_dct_blocks_per_thread_block / GPUJPEG_BLOCK_SIZE;

        const dim3 num_kernel_blocks = ceiling_div(num_blocks_c, num_elements_per_block);
        const dim3 kernel_block_size =
            dim3(GPUJPEG_IDCT_BLOCK_X, GPUJPEG_IDCT_BLOCK_Y, GPUJPEG_IDCT_BLOCK_Z);

        gpujpeg_idct_gpu_kernel<<<num_kernel_blocks, kernel_block_size, 0, stream>>>(
            coeffs[c].ptr, pixels[c].ptr, qtable[c].ptr, num_blocks_c);
        RETURN_IF_ERR_CUDA(cudaPeekAtLastError());
    }

    return true;
}
