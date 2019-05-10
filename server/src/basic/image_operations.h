#ifndef SERVER_BASIC_IMAGE_OPERATIONS_H_
#define SERVER_BASIC_IMAGE_OPERATIONS_H_

#include <opencv2/core/core.hpp>
#include <saliency/some_definition.h>


//cmcv
namespace imageoperations{
class ImageOperations {
 public:
  // Get continuous None-Zero label Region with Largest Sum value
  static cv::Mat GetLargestSumNoneZeroRegion(const cv::Mat& mask1u,
                                             double ignore_ratio = 0.02);
  // Get mask region.
  static cv::Rect GetMaskRange(const cv::Mat& mask1u,
                               int ext = 0, int thresh = 10);

  // Get continuous components for non-zero labels. Return region index mat (region index
  // of each mat position) and sum of label values in each region
  static int GetNoneZeroRegions(const cv::Mat_<byte>& label1u,
                                cv::Mat_<int>& regIdx1i, vector<int>& idxSum);
};


}

#endif
