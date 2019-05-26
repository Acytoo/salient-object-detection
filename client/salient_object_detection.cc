

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
  double precision=0.0, recall=0.0, f=0.0;
  int runtime=0, amount=0;
  regioncontrast::RegionContrast::ProcessImages(image_folder, amount, runtime,
                                                true, precision, recall, f);
  cout << precision << "    " << recall << "    " << f << "    " << runtime << endl;
  return 0;
}
