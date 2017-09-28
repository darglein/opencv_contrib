/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_CUDAARITHM


#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/core.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

namespace cv { namespace cuda {

__constant__ float d_Kernel[21];


template<typename T>
__CV_CUDA_HOST_DEVICE__
T& clampedRead(PtrStepSz<T> src, int y, int x)
{
    x = ::min(::max(0,x),src.cols-1);
    y = ::min(::max(0,y),src.rows-1);
    return src(y,x);
}

template<typename T>
__CV_CUDA_HOST_DEVICE__
void clampedWrite(PtrStepSz<T> src, int y, int x, T v)
{
    if(x >= 0 && x < src.cols && y >=0 && y < src.rows)
        src(y,x) = v;
}



template<typename T, int RADIUS, unsigned int BLOCK_W, unsigned int BLOCK_H, unsigned int Y_ELEMENTS>
__global__ static
void d_convolveInner(PtrStepSz<T> src, PtrStepSz<T> dst)
{
    const unsigned int TILE_H = BLOCK_H;
    const unsigned int TILE_W = BLOCK_W;

    const unsigned int TILE_H2 = TILE_H * Y_ELEMENTS;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    //    int t = tx + ty * BLOCK_W;

    int x_tile = blockIdx.x * (TILE_W - 2 * RADIUS) - RADIUS;
    int y_tile = blockIdx.y * (TILE_H2 - 2 * RADIUS) - RADIUS;

    int x = x_tile + tx;
    int y = y_tile + ty;


    __shared__ T buffer[TILE_H2][TILE_W];
    __shared__ T buffer2[TILE_H2 - RADIUS * 2][TILE_W];



    //copy main data
    for(int i = 0; i < Y_ELEMENTS; ++i)
        //        buffer[ty + i * TILE_H][tx]  = src.clampedRead(y + i * TILE_H,x);
        buffer[ty + i * TILE_H][tx]  = clampedRead(src,y + i * TILE_H,x);



    __syncthreads();


    T *kernel = d_Kernel;

    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        //        int gx = x;
        //        int gy = y + i * TILE_H;
        int lx = tx;
        int ly = ty + i * TILE_H;

        if(ly < RADIUS || ly >= TILE_H2 - RADIUS)
            continue;

        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer[ly + j][lx] * kernel[kernelIndex];
        }
        buffer2[ly - RADIUS][lx] = sum;
    }



    __syncthreads();

    for(int i = 0; i < Y_ELEMENTS; ++i)
    {
        int gx = x;
        int gy = y + i * TILE_H;

        int lx = tx;
        int ly = ty + i * TILE_H;

        if(ly < RADIUS || ly >= TILE_H2 - RADIUS)
            continue;

        if(lx < RADIUS || lx >= TILE_W - RADIUS)
            continue;

        T sum = 0;
#pragma unroll
        for (int j=-RADIUS;j<=RADIUS;j++)
        {
            int kernelIndex = j + RADIUS;
            sum += buffer2[ly - RADIUS][lx + j] * kernel[kernelIndex];
        }


        clampedWrite(dst,gy,gx,sum);
    }

}

template<typename T, int RADIUS, bool LOW_OCC>
inline
void convolveInner(PtrStepSz<T> src, PtrStepSz<T> dst){
    int w = src.cols;
    int h = src.rows;


    const int BLOCK_W = LOW_OCC ? 64 : 32;
    const int BLOCK_H = LOW_OCC ? 8 : 16;
    const int Y_ELEMENTS = LOW_OCC ? 4 : 2;
    dim3 blocks(
                divUp(w, BLOCK_W - 2 * RADIUS),
                divUp(h, BLOCK_H * Y_ELEMENTS - 2 * RADIUS),
                1
                );

    //    dim3 blocks(Saiga::CUDA::getBlockCount(w, BLOCK_W), Saiga::CUDA::getBlockCount(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);

    d_convolveInner<T,RADIUS,BLOCK_W,BLOCK_H,Y_ELEMENTS> <<<blocks, threads>>>(src,dst);
}


void convolveSinglePassSeparateInner75(SiftImageType src, SiftImageType dst, GpuMat& kernel){

    int size = kernel.rows * kernel.cols;
    int radius = size / 2;
    cudaMemcpyToSymbol(d_Kernel, kernel.data, size*sizeof(float),0,cudaMemcpyDeviceToDevice);
    switch (radius){
    case 1: convolveInner<float,1,true>(src,dst); break;
    case 2: convolveInner<float,2,true>(src,dst); break;
    case 3: convolveInner<float,3,true>(src,dst); break;
    case 4: convolveInner<float,4,true>(src,dst); break;
    case 5: convolveInner<float,5,true>(src,dst); break;
    case 6: convolveInner<float,6,true>(src,dst); break;
    case 7: convolveInner<float,7,true>(src,dst); break;
    case 8: convolveInner<float,8,true>(src,dst); break;
    }
}




static texture<float, cudaTextureType2D, cudaReadModeElementType> floatTex;


template<int BLOCK_W, int BLOCK_H, int ROWS_PER_THREAD>
__global__
static void d_scaleUp2Linear(SiftImageType src, SiftImageType dst, int h, double scale_x, double scale_y)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x*BLOCK_W + tx;
    int y = blockIdx.y*BLOCK_H + ty;

    if(x >= dst.cols)
        return;

#pragma unroll
    for(int i = 0; i < ROWS_PER_THREAD; ++i, y+=h){
        if(y < dst.rows){
            //use hardware bil. interpolation
            float xf = (float(x) + 0.5f) * scale_x;
            float yf = (float(y) + 0.5f) * scale_y;
            dst(y,x) = tex2D(floatTex,xf,yf);


        }
    }

}


void scaleUp2Linear(SiftImageType src, SiftImageType dst){

    textureReference& floatTexRef = floatTex;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
    size_t offset;
    //    SAIGA_ASSERT(src.step % 256 == 0);
    cudaBindTexture2D(&offset, &floatTexRef, src.data, &desc, src.cols, src.rows, src.step);

    floatTexRef.addressMode[0] = cudaAddressModeClamp;
    floatTexRef.addressMode[1] = cudaAddressModeClamp;
    floatTexRef.filterMode = cudaFilterModeLinear;
    floatTexRef.normalized = false;




    double inv_scale_x = (double)dst.cols/src.cols;
    double inv_scale_y = (double)dst.rows/src.rows;
    double scale_x = 1./inv_scale_x, scale_y = 1./inv_scale_y;


    const int ROWS_PER_THREAD = 4;
    const int BLOCK_W = 128;
    const int BLOCK_H = 1;
    int w = dst.cols;
    int h = divUp(dst.rows,ROWS_PER_THREAD);
    dim3 blocks(divUp(w, BLOCK_W), divUp(h, BLOCK_H));
    dim3 threads(BLOCK_W, BLOCK_H);
    d_scaleUp2Linear<BLOCK_W,BLOCK_H,ROWS_PER_THREAD> <<<blocks, threads>>>(src,dst,h,scale_x,scale_y);
}


void createInitialImage(SiftImageType src, SiftImageType dst, SiftImageType tmp, GpuMat& initialBlurKernel, bool doubleScale){
    if (!doubleScale) {
        convolveSinglePassSeparateInner75(src,dst,initialBlurKernel);
    }else{
        //note: the blur takes up roughly 2x the time of scale up
        scaleUp2Linear(src,tmp);
        convolveSinglePassSeparateInner75(tmp,dst,initialBlurKernel);
    }

}



}} // namespace cv { namespace cuda { namespace cudev

#endif // HAVE_OPENCV_CUDAARITHM
