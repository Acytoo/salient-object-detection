#include <basic/segment_image.h>

#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void write_csv(string filename, Mat m) {
  ofstream myfile;
  myfile.open(filename.c_str());
  myfile<< cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
  myfile.close();
}

int main(int argc, char* argv[]) {
  string img_path = "/home/acytoo/Pictures/coco_stop.jpg";
  if (argc == 2){
    img_path = argv[1];
  }

  Mat img3u = imread(img_path, 1);
  Mat img_index, img3f, img_lab3f;
  img3u.convertTo(img3f, CV_32FC3, 1.0/255);
  cvtColor(img3f, img_lab3f, CV_BGR2Lab);
  int region_num = SegmentImage(img_lab3f, img_index);
  Mat img_label3u;
  int err = ShowLabel(img_index, img_label3u, region_num);
  if (err) {
    cout << "error\n";
    return -1;
  }
  write_csv("img_index.csv", img_index);
  cout << img_index << endl << endl << endl;
  cout << "region number: " << region_num << endl;
  cout << "index size: " << img_index.size() << endl;
  cout << "ori img size: " << img3u.size() << endl;
  imshow("ori", img3u);
  imshow("segment", img_label3u);
  waitKey(0);
  return 0;
}
