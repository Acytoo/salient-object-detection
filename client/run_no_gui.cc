#include <saliency/saliency_cut.h>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string img_path = "/home/acytoo/Pictures/COCO_stop.jpg";
  string result_path = "";
  // string img_path = "/home/acytoo/Pictures/nfr.png";
  // string img_path = "/home/acytoo/Pictures/white_black.jpg";
  // string img_path = "/home/acytoo/Pictures/2colors.png";
  // string img_path = "/home/acytoo/Pictures/hotblack.jpg";
  // string img_path = "/home/acytoo/Pictures/black.png";
  // jpg has 3 channels, and bmp sometimes has 4, in color
  saliencycut::SaliencyCut::ProcessSingleImg(img_path, result_path);
  cout << result_path << endl;
  return 0;
}
