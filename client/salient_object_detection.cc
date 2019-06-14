

#include <saliency/saliency_region_contrast.h>

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {
  // string img_path = "/home/acytoo/Pictures/6112.jpg";
  // Mat ori_img3u = imread(img_path);
  // imshow("ori", ori_img3u);
  // string res_salient, res_salient_bi, res_salient_cut;
  // regioncontrast::RegionContrast::ProcessSingleImg(img_path, res_salient, res_salient_bi, res_salient_cut);


  string image_folder = "/home/acytoo/workSpace/salient-object-detection/data/saliency_test";
  double precision=0.0, recall=0.0, f=0.0, cut_threshold = 1.55;
  int runtime=0, amount=0;
  vector<double> cut_thresholds = {0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95};
  // vector<double> cut_thresholds = {2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0};

  for (auto i : cut_thresholds)
    regioncontrast::RegionContrast::ProcessImages(image_folder, amount, runtime,
                                                true, precision, recall, f, i);
  // cout << precision << "    " << recall << "    " << f << "    " << runtime << endl;

  // Mat img3f = imread("/home/acytoo/Pictures/195780.jpg");
  // img3f.convertTo(img3f, CV_32FC3, 1.0/255, 0);
  // Mat salni = imread("/home/acytoo/Pictures/195780.png", 0);
  // salni /= 255;
  // Mat res;
  // regioncontrast::RegionContrast::CutImage(img3f, salni, res);
  // imshow("Res", res*255);
  // waitKey(0);
  // imwrite("/home/acytoo/Pictures/195780_color.png", res*255);
  return 0;
}
