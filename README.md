# GPUDCT

The goal of this project is to implement inverse discrete cosine transform (IDCT) for JPEG decoding, achieving the highest possible performance while staying within acceptable error margins. Specific goals are:

- Target hardware is the NVIDIA RTX 2070, as I have this at my disposal.
- Error is calculated as the peak signal-to-noise ratio (PSNR), following previous work<sup>[1](#references)</sup>. PSNR values of 50 or higher are deemed acceptable.
- For use in JPEG decoding, the implementation should load 16-bits integer coefficient values and write 8-bits pixel values. The implementation should not only perform IDCT, but also dequantization using 16-bit quantization values.

## Benchmark

A benchmark is implemented that accepts a JPEG image and if none is provided, runs a synthetic benchmark:

```shell
sudo ./gpudct_benchmark
synthetic benchmark mode.. finished generating random coefficients
memory read bandwidth: 208.646059 GiB/s 224.032000 GB/s
                 avg [    min,   max] GiB/s        avg ms
       naive:   1.76 [  1.60,   1.79] GiB/s 1138.98804 ms
         lut:  10.39 [  9.50,  10.73] GiB/s  192.51953 ms
   seperable:  33.15 [ 30.85,  34.79] GiB/s   60.32524 ms
  decomposed: 125.49 [121.00, 138.31] GiB/s   15.93758 ms
   no_shared: 197.96 [174.83, 217.59] GiB/s   10.10308 ms
        next: 216.51 [183.07, 234.95] GiB/s    9.23755 ms
      next16: 226.23 [192.95, 242.02] GiB/s    8.84045 ms
     gpujpeg: 119.81 [112.23, 130.19] GiB/s   16.69348 ms
```

The benchmark locks GPU clocks for consistent performance, if run with required permissions. The implementations are explained [below](#implementations).

Since the kernels load one 16-bits value (and quantization value) per pixel and store one 8-bits value per pixel, the kernel will be limited by read bandwidth, not write bandwidth. On the Turing architecture, the read and write bandwidth are equal and 209 GiB/s (224 GB/s). Since they can be performed simultaneously, the effective bandwidth is 418 GiB/s (448 GB/s). However, for the IDCT application the bottleneck is read bandwidth.

The final non-reference implementation (`next16`) achieves performance at or above bandwidth, meaning likely no substantial improvement can further be obtained. Why bandwidth higher than theoretically possible is obtained is unclear, perhaps there is some memory clock boost.

## Test

A simple test is implemented that calculates the PSNR values of the image components (`006mp-cathedral.jpg` from jpeg-test-images<sup>[2](#references)</sup>):

```shell
./gpudct_test 006mp-cathedral.jpg
       naive [108.934181, 82.503609, 82.689301]
         lut [108.934181, 81.949699, 81.444550]
   seperable [108.934181, 81.949699, 81.444550]
  decomposed [ 83.799339, 76.518715, 77.617996]
   no shared [ 83.799339, 76.518715, 77.617996]
        next [ 66.770836, 78.647659, 81.513214]
      next16 [ 66.770836, 78.647659, 81.513214]
     gpujpeg [ 66.906792, 63.323742, 60.480869]
```

The PSNR values on all implementations stay well within the acceptable value of 50 (for this particular image). The implementations are explained [below](#implementations).

## Implementations

The implementations are as follows:
- `naive` is a direct implementation and calculates the cosines at runtime. Each thread calculates a single pixel value by summing 8 x 8 = 64 coefficients and factors.
- `lut` uses a look-up table for the cosines.
- `seperable` separates the computation and stores intermediate results in shared memory. Each thread still computes a single pixel value, but instead only sums 8 + 8 = 16 coefficients and factors.
- `decomposed` employs the algorithm of Sung et al.<sup>[3](#references)</sup> that decomposes the IDCT algorithm into eight per-row equations and subsequently eight per-column equations. Inspired by the CUDA samples<sup>[1](#references)</sup>, eight threads process one block of 64 coefficients. Between computations, the coefficients is transposed in shared memory.
- `no shared` transposes the coefficients using warp-level intrinsics instead of shared memory.
- `next` makes micro-optimizations to the previous kernel.
- `next16` assigns sixteen coefficients rather than eight coefficients to each thread, allowing to transpose the coefficients between computations with fewer total instructions.
- `gpujpeg` is a reference implementation from GPUJPEG<sup>[4](#references)</sup> with minor adjustments to process the data format.

While nice reference material, the kernels from the CUDA samples<sup>[1](#references)</sup> are not included in the benchmark and test comparisons. The reason is that quantization is not done within those IDCT kernels, and those kernels expect the coefficients to be in raster order. As a result, too many modifications would be required to include them, causing them to no longer be useful performance references.

## References

1. Discrete Cosine Transform for 8x8 Blocks with CUDA, Anton Obukhov and Alexander Kharlamov, 2008 ([download](https://developer.download.nvidia.com/assets/cuda/files/dct8x8.pdf), [code](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/dct8x8)).
2. JPEG test images, Nol Moonen [code](https://github.com/nolmoonen/jpeg-test-images).
3. High-Efficiency and Low-Power Architectures for 2-D DCT and IDCT Based on CORDIC Rotation, Sung et al., 2006 ([reference](https://ieeexplore.ieee.org/document/4032176)).
4. GPUJPEG, CESNET [code](https://github.com/CESNET/GPUJPEG).
