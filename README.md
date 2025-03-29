# GPUDCT

## Project goals

The goal is to implement a CUDA kernels that obtain the highest performance possible while staying within a reasonable error margin. Performance will be measured on an RTX 2070 (because I have that at my disposal). The error will be measured with PSNR because it is simple and used in prior art.

- Coefficients are in raster order. Because that is how they are returned by libjpeg.
- Qtable is given in raster order. Because that is how it is returned by libjpeg and because it easy to put in raster order without little overhead.
- PSNR is calculated per block because that is where error occurs, this allows for the smallest fp errors. it should be higher thatn 50 with some margin? depends on measurements
- run as fast as possible on rtx 2070 because I have that hardware. 
- use the test images. take into account image size when benchmarking, not to save entirely in cache
- put coefficients in one buffer, to not have launch overhead. no, differen qtables
- make note that psnr testing approach might not be the greatest for syntethic images
- enable --fast-math by default but maybe not for test as it seems to mess up the psnr values quite significantly

stretch goals
- fdct
- cpu simd version
- make cpu_buf to mirror, proper error handling on null malloc
- add syntethic test and benchmark?

steps in optimization are:
- naive version
- lut for cosine-ish coefficients
- seperable computation
- loading and storing optimizaiton
- micro optimization?

todo
- maybe do not use maximum psnr but normal psnr, as this version is very sensitive to minor changes. the problem is that the order of the multiplications of `qval`, `coeff`, and constants starts to matter.

tried but didn;t work
- qtable in constant memory
- not doing final transpose but writing uncoalesced u8 or u16
