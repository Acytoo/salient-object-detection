#include "segment_image.h"

#include <iostream>
// demo for some basic operations during salieny- cut
namespace basic_functions_demo {
  int SegmentDemo(const std::string& ori_path, std::string& res_path);

  int Bgr2labDemo(const std::string& ori_path, std::string& res_path);

  int QuantizeDemo(const std::string& ori_path, std::string& res_path);

  int WriteCsv(const std::string& filename, const cv::Mat& m);

}
