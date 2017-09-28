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
#include <iostream>
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

void createInitialImage(SiftImageType src, SiftImageType dst, SiftImageType tmp, GpuMat& initialBlurKernel, bool doubleScale);

void testWrite(std::string path, SiftImageType img)
{
//    GpuMat test;
//    test.data = (uchar*)img.data;
//    test.cols = img.cols;
//    test.rows = img.rows;
//    test.step = img.step;
//    test.type


    std::cout << img.rows << " " << img.cols << " " << img.step << std::endl;
      GpuMat test(img.rows,img.cols, CV_32F,img.data, img.step);

    Mat m;
    test.download(m);
    imwrite(path,m);
}

int SIFT_CUDA::compute(const GpuMat& img, GpuMat& keypoints, GpuMat& descriptors) {
    initMemory();

//    CV_ASSERT(img.type() == CV_32F);
    std::cout << img.type() << std::endl;
//    Mat m;
//    img.download(m);
//    imwrite("out/init.jpg",m);

    testWrite("out/init.jpg",img);

    createInitialImage(img,gaussianPyramid2[0],gaussianPyramid2[1],initialBlurKernel,doubleScale);


    testWrite("out/blurredinitial.jpg",gaussianPyramid2[0]);
//    GpuMat test;

//    img.download(m);
//    imwrite("out/blurredinitial.jpg",m);

    return 0;
}



}} // namespace cv { namespace cuda { namespace cudev

#endif // HAVE_OPENCV_CUDAARITHM
