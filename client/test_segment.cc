#include <basic/segment_image.h>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  string img_path = "/home/acytoo/Pictures/coco_stop.jpg";
  if (argc == 2){
    img_path = argv[1];
  }

  Mat img3u = imread(img_path, 1);
  imshow("ori", img3u);
  Mat img_index, img3f, img_lab3f;
  img3u.convertTo(img3f, CV_32FC3, 1.0/255);
  //
  cvtColor(img3f, img_lab3f, CV_BGR2Lab);
  int region_num = SegmentImage(img_lab3f, img_index);
  Mat img_label3u;
  int err = ShowLabel(img_index, img_label3u, region_num);
  if (err) {
    cout << "error\n";
  }
  imshow("segment", img_label3u);
  waitKey(0);
  return 0;
}
