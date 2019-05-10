#include "basic_functions_demo.h"

#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <basic/segment_image.h>
#include <saliency/saliency_region_contrast.h>

namespace basic_functions_demo {

inline int WriteCsv(const string& filename, const cv::Mat& m) {
  ofstream myfile;
  myfile.open(filename.c_str());
  myfile<< cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
  myfile.close();
  return 0;
}

int SegmentDemo(const string& ori_path, string& res_path) {
  int end_pos = ori_path.rfind(".");
  res_path = ori_path.substr(0, end_pos) + "_seg.png";
  Mat img3u = imread(ori_path, 1);
  if (!img3u.data) {
    // empty
    return -1;
  }
  Mat img_index, img3f, img_lab3f;
  img3u.convertTo(img3f, CV_32FC3, 1.0/255);
  cvtColor(img3f, img_lab3f, COLOR_BGR2Lab);
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


int Bgr2labDemo(const std::string &ori_path, std::string &res_path) {
  int end_pos = ori_path.rfind(".");
  res_path = ori_path.substr(0, end_pos) + "_lab.png";
  Mat img3u = imread(ori_path, 1);
  if (!img3u.data) {
    // empty
    return -1;
  }
  Mat img_lab3u;
  cvtColor(img3u, img_lab3u, COLOR_BGR2Lab);
  imwrite(res_path, img_lab3u);
  return 0;
}

int QuantizeDemo(const std::string& ori_path, std::string& res_path) {
  int end_pos = ori_path.rfind(".");
  res_path = ori_path.substr(0, end_pos) + "_quantize.png";
  Mat img3u = imread(ori_path, 1);
  if (!img3u.data) {
    // empty
    return -1;
  }

  Mat colorIdx1i, img3f, tmp, color3fv, reg_sal1v;
  img3u.convertTo(img3f, CV_32FC3, 1.0/255);
  const int tempint[3] = {12,12,12};
  int quantize_num = regioncontrast::RegionContrast::Quantize(img3f, colorIdx1i, color3fv, tmp, 0.95, tempint);
  cout << "quantize number " << quantize_num << endl;
  if (quantize_num == 2){
    printf("quantize_num == 2, %d: %s\n", __LINE__, __FILE__);
    Mat sal;
    cout << colorIdx1i.type() << " " << colorIdx1i.size() << endl;
    WriteCsv("colorIndex1i.csv", colorIdx1i);
    compare(colorIdx1i, 1, sal, CMP_EQ);
    sal.convertTo(sal, CV_32F, 1.0/255);
    cout << sal.type() << " " << sal.size() << endl;
    imwrite(res_path, sal);
    WriteCsv("sali.csv", sal);
    return -2;
  }
  if (quantize_num <= 1) {
    printf("quantize number     1\n");
    return -1;
  }

  imwrite(res_path, color3fv);



  return -3;
}

// end namespace: basic_functions_demo
}
