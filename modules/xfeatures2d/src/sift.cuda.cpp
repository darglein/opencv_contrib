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

#include "precomp.hpp"
#include "sift_defines.h"
#include <iostream>




#if 1

using namespace cv;
using namespace cv::cuda;

#if (!defined (HAVE_CUDA) || !defined (HAVE_OPENCV_CUDAARITHM))



#else // !defined (HAVE_CUDA)



static cuda::GpuMat createGaussianBlurKernel(int radius, float sigma){
    const int ELEMENTS = radius * 2 + 1;
    std::vector<float> kernel(ELEMENTS);
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*sigma*sigma);
    for (int j=-radius;j<=radius;j++) {
        kernel[j+radius] = (float)expf(-(double)j*j*ivar2);
        kernelSum += kernel[j+radius];
    }
    for (int j=-radius;j<=radius;j++)
        kernel[j+radius] /= kernelSum;
    return cuda::GpuMat(kernel);
}



SIFT_CUDA::SIFT_CUDA(int _imageWidth, int _imageHeight, bool _doubleScale, int maxOctaves,
                     int _nfeatures, int _nOctaveLayers,
                     double _contrastThreshold, double _edgeThreshold, double _sigma )
    : imageWidth(_imageWidth),imageHeight(_imageHeight),doubleScale(_doubleScale),
      nfeatures(_nfeatures),nOctaveLayers(_nOctaveLayers),contrastThreshold(_contrastThreshold),edgeThreshold(_edgeThreshold),sigma(_sigma),initialized(false)
{
    numOctaves =  cvRound(std::log( (double)std::min( imageWidth, imageHeight ) ) / std::log(2.) - 2) + 1;
    if(maxOctaves > 0)
        numOctaves = std::min(numOctaves,maxOctaves);
}



void SIFT_CUDA::initMemory()
{
    if(initialized)
        return;
#ifdef SIFT_PRINT_TIMINGS
    Saiga::CUDA::CudaScopedTimerPrint tim("SIFT_CUDA::initMemory");
#endif

    gaussianPyramid2.resize(numOctaves * (nOctaveLayers + 3));
    dogPyramid2.resize(numOctaves * (nOctaveLayers + 2));

//    pointCounter.resize(1);
    pointCounter.create(1,1,CV_32S);

#ifdef SIFT_DEBUG
    std::cout << " ==== ExtractSift nOctaves=" << numOctaves << " octave layers: " << nOctaveLayers << std::endl;
#endif
    int width = imageWidth*(doubleScale ? 2 : 1);
    int height = imageHeight*(doubleScale ? 2 : 1);

    //size of the gaussian pyramid in float
    size_t gpyramidSize = 0;
    size_t dpyramidSize = 0;
    size_t tmpSize = 0;
    //image row alignment
    const int alignment = 256;

    for (int o=0, w = width, h = height; o<numOctaves; o++) {
//        int pitch = Saiga::iAlignUp(w * sizeof(float), alignment);
        int pitch = cvAlign(w * sizeof(float), alignment);
        size_t imageSize = h * pitch;
        gpyramidSize += (nOctaveLayers + 3) * imageSize;
        dpyramidSize += (nOctaveLayers + 2) * imageSize;
        tmpSize += imageSize;
#ifdef SIFT_DEBUG
        cout << "Octave " << o << " - ImageSize: " << w << "x" << h << ", Pitch: " <<  pitch << "x" << h  << ", MemoryPerImage: " << imageSize << ", MemoryPerOctave: " << imageSize*(nOctaveLayers + 3)  << endl;
#endif
        w /= 2;
        h /= 2;
    }


//    memorygpyramid.resize(gpyramidSize);
//    memorydogpyramid.resize(dpyramidSize);

    memorygpyramid.create(gpyramidSize,1,CV_8S);
    memorydogpyramid.create(dpyramidSize,1,CV_8S);



    size_t ps = 0;
    size_t dps = 0;

    for (int o=0, w = width, h = height; o<numOctaves; o++) {
        int pitch = cvAlign(w * sizeof(float), alignment);
        size_t imageSize = h * pitch;

        for(int j = 0; j < nOctaveLayers + 3 ; ++j){
            int index = o * (nOctaveLayers + 3) + j;
            gaussianPyramid2[index] = SiftImageType(h,w, (float*)(memorygpyramid.data + ps),pitch) ;
            ps += imageSize;
        }

        for(int j = 0; j < nOctaveLayers + 2 ; ++j){
            int index = o * (nOctaveLayers + 2) + j;
            dogPyramid2[index] = SiftImageType(h,w,(float*)(memorydogpyramid.data+dps),pitch
                                               );
            dps += imageSize;
        }

        w /= 2;
        h /= 2;
    }

    if (!doubleScale) {
        float sig_diff = sqrtf( std::max<double>(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f) );
        initialBlurKernel = createGaussianBlurKernel(GAUSSIAN_KERNEL_RADIUS,sig_diff);
    }else{
        float sig_diff = sqrtf( std::max<double>(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f) );
        initialBlurKernel = createGaussianBlurKernel(GAUSSIAN_KERNEL_RADIUS,sig_diff);
    }

    std::vector<double> sig(nOctaveLayers + 3);
    // precompute Gaussian sigmas using the following formula:
    //  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
    sig[0] = sigma;
    double k = std::pow( 2., 1. / nOctaveLayers );
    for( int i = 1; i < nOctaveLayers + 3; i++ )
    {
        double sig_prev = std::pow(k, (double)(i-1))*sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }
    octaveBlurKernels.resize(nOctaveLayers + 3);

    for( int i = 0; i < nOctaveLayers + 3; i++ )
    {
        octaveBlurKernels[i] = createGaussianBlurKernel(GAUSSIAN_KERNEL_RADIUS,sig[i]);
    }

    std::cout << "memory initialized :D" << std::endl;
    initialized = true;
}

#endif // !defined (HAVE_CUDA)
#endif

