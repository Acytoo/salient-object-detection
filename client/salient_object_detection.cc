#include <saliency/saliency_cut.h>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string img_path = "/home/acytoo/Pictures/COCO_stop.jpg";
  // cout << "imput the path of your image file" << endl;
  // cin >> img_path;
  string result_rc_path = "", result_rcc_path = "";
  // string img_path = "/home/acytoo/Pictures/nfr.png";
  // string img_path = "/home/acytoo/Pictures/white_black.jpg";
  // string img_path = "/home/acytoo/Pictures/2colors.png";
  // string img_path = "/home/acytoo/Pictures/hotblack.jpg";
  // string img_path = "/home/acytoo/Pictures/black.png";
  // jpg has 3 channels, and bmp sometimes has 4, in color
  // saliencycut::SaliencyCut::ProcessSingleImg(img_path, result_rc_path,
  //                                            result_rcc_path);

  string root_dir_path = "./test_img";
  int number = saliencycut::SaliencyCut::ProcessImages(root_dir_path);
  cout << number << endl;

  // cout << result_rc_path << endl;
  // cout << result_rcc_path << endl;
  return 0;
}
