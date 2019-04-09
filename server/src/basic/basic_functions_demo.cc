#include "basic_functions_demo.h"


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <basic/segment_image.h>

int basic_functions_demo::SegmentDemo(const string& ori_path, string& res_path) {
  int end_pos = ori_path.rfind(".");
  res_path = ori_path.substr(0, end_pos) + "_seg.png";
  Mat img3u = imread(ori_path, 1);
  if (!img3u.data) {
    // empty
    return -1;
  }
  Mat img_index, img3f, img_lab3f;
  img3u.convertTo(img3f, CV_32FC3, 1.0/255);
  cvtColor(img3f, img_lab3f, CV_BGR2Lab);
  int region_num = SegmentImage(img_lab3f, img_index);
  Mat img_label3u;
  int err = ShowLabel(img_index, img_label3u, region_num);
  if (err) {
    // cout << "error\n";
    return -2;
  }
  imwrite(res_path, img_label3u);
  return 0;
}


int basic_functions_demo::Bgr2labDemo(const std::string &ori_path, std::string &res_path) {
  int end_pos = ori_path.rfind(".");
  res_path = ori_path.substr(0, end_pos) + "_lab.png";
  Mat img3u = imread(ori_path, 1);
  if (!img3u.data) {
    // empty
    return -1;
  }
  Mat img_lab3u;
  cvtColor(img3u, img_lab3u, CV_BGR2Lab);
  imwrite(res_path, img_lab3u);
  return 0;
}
