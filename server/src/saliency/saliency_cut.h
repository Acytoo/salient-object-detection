#ifndef SERVER_SALIENCY_SALIENCY_CUT_H_
#define SERVER_SALIENCY_SALIENCY_CUT_H_

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>

namespace saliencycut {

  class SaliencyCut {
  public:
    static int ProcessSingleImg(const std::string& img_path,
                                std::string& result_path);
    static void ShowImageInfo(const cv::Mat& img);



  };
}

#endif  // SERVER_SALIENCY_SALIENCY_CUT_H_
