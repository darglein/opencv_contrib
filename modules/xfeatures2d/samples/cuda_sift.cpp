#include <iostream>
#include <stdio.h>
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

using namespace cv;
using namespace cv::xfeatures2d;

using std::cout;
using std::endl;

void computeCPU(std::string file){

    Mat img1;


    imread(file, IMREAD_GRAYSCALE).copyTo(img1);


    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    //cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
    //cv::Ptr<Feature2D> f2d = ORB::create();
    // you get the picture, i hope..

    //-- Step 1: Detect the keypoints:
    std::vector<KeyPoint> keypoints;
    f2d->detect( img1, keypoints );

    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1;
    f2d->compute( img1, keypoints, descriptors_1 );


    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(img1, keypoints, output, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imwrite("sift_result.jpg", output);


    //    namedWindow("surf matches", 0);
    //    imshow("surf matches", img1);

    imwrite("test.jpg", output);
}


void computeGPU(std::string file){

    Mat1f img1;
    imread(file, IMREAD_GRAYSCALE).copyTo(img1);



   if(img1.step % 256 != 0)
   {
       std::cout << "failed" << endl;
       return;
   }



    cuda::SIFT_CUDA sc(img1.cols,img1.rows,true,-1,10000);
    sc.initMemory();

    cuda::GpuMat imggpu(img1);

    cuda::GpuMat d_keypoints, d_descriptors;
    sc.compute(imggpu,d_keypoints,d_descriptors);


    // Add results to image and save.
//    cv::Mat output;
//    cv::drawKeypoints(img1, keypoints, output, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    cv::imwrite("sift_result.jpg", output);


    //    namedWindow("surf matches", 0);
    //    imshow("surf matches", img1);

//    imwrite("test.jpg", output);
}

int main(int argc, char* argv[])
{

    std::string leftName = "tsukuba.png";
//    computeCPU(leftName);


    computeGPU(leftName);
    cout << "finished" << endl;
    //    waitKey(0);
    return EXIT_SUCCESS;
}

