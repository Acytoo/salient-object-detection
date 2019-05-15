

#include <saliency/saliency_region_contrast.h>

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
  string img_path = "/home/acytoo/Pictures/12.bmp";
  // cout << "imput the path of your image file" << endl;
  // cin >> img_path;
  Mat ori_img3u = imread(img_path);
  imshow("ori", ori_img3u);
  string result_rc_path = "";
  // string img_path = "/home/acytoo/Pictures/nfr.png";
  // string img_path = "/home/acytoo/Pictures/white_black.jpg";
  // string img_path = "/home/acytoo/Pictures/2colors.png";
  // string img_path = "/home/acytoo/Pictures/hotblack.jpg";
  // string img_path = "/home/acytoo/Pictures/black.png";
  // jpg has 3 channels, and bmp sometimes has 4, in color
  regioncontrast::RegionContrast::ProcessSingleImg(img_path, result_rc_path);

  // string root_dir_path = "./test_img";
  // int amount = 0, time_cost = 0;
  // int number = saliencycut::SaliencyCut::ProcessImages(root_dir_path, amount, time_cost);
  // cout << number << endl;

  // cout << result_rc_path << endl;
  // cout << result_rcc_path << endl;
  return 0;
}
