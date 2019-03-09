#include "saliency_cut.h"
#include <saliency/saliency_region_contrast.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <stdio.h>




using namespace std;
using namespace cv;

int saliencycut::SaliencyCut::ProcessSingleImg(const string& img_path,
                                               string& result_path) {

  int end_pos = img_path.rfind(".");
  result_path = img_path.substr(0, end_pos) + "_RC.png";
  //first error detection, permission and presense of the image
  //imread 2nd parameter 0: grayscale; 1: 3 channels; -1: as is(with alpha channel)
  Mat img3f = imread(img_path, 1);
  //convert to float, 3 channels
  img3f.convertTo(img3f, CV_32FC3, 1.0/255, 0);


  Mat sal = regioncontrast::RegionContrast::GetRegionContrast(img3f);
  // imshow("sal", sal);
  // waitKey(0);
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  imwrite(result_path, sal*255, compression_params);
  cout << "finish" << endl;

  // finish region based saliency region detection

  return 0;
}


void saliencycut::SaliencyCut::ShowImageInfo(const Mat& img) {
  // +--------+----+----+----+----+------+------+------+------+
  //   |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
  //   +--------+----+----+----+----+------+------+------+------+
  //   | CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
  //   | CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
  //   | CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
  //   | CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
  //   | CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
  //   | CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
  //   | CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
  //   +--------+----+----+----+----+------+------+------+------+

  cout << "M = "<< endl << " "  << img << endl << endl;

  std::cout << " channels " << img.channels()
            << " type " << img.type()
            << " size " << img.size()
            << std::endl;



  // cv::Vec3b vec3b = img.at<cv::Vec3b>(0,0);
  // uchar vec3b0 = img.at<cv::Vec3b>(0,0)[0];
  // uchar vec3b1 = img.at<cv::Vec3b>(0,0)[1];
  // uchar vec3b2 = img.at<cv::Vec3b>(0,0)[2];
  // std::cout<<"vec3b = "<<vec3b<<std::endl;
  // std::cout<<"vec3b0 = "<<(int)vec3b0<<std::endl;
  // std::cout<<"vec3b1 = "<<(int)vec3b1<<std::endl;
  // std::cout<<"vec3b2 = "<<(int)vec3b2<<std::endl;

  cv::Vec3d vec3d = img.at<cv::Vec3d>(0,0);
  double vec3d0 = img.at<cv::Vec3d>(0,0)[0];
  double vec3d1 = img.at<cv::Vec3d>(0,0)[1];
  double vec3d2 = img.at<cv::Vec3d>(0,0)[2];
  std::cout<<"vec3d = "<<vec3d<<std::endl;

}
