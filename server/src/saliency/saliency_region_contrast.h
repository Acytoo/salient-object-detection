#ifndef SERVER_SALIENCY_SALIENCY_REGION_CONTRAST_H_
#define SERVER_SALIENCY_SALIENCY_REGION_CONTRAST_H_

#include <saliency/some_definition.h>

namespace regioncontrast{
class RegionContrast{
 public:
  static cv::Mat GetRegionContrast(const cv::Mat& img3f);

  static void SmoothByHist(const cv::Mat& img3f, cv::Mat& sal1f, float delta);

  static void SmoothByRegion(cv::Mat& sal1f, const cv::Mat& idx1i,
                             int regNum, bool bNormalize = true);

  static int Quantize(const cv::Mat &img3f, cv::Mat &color_idx1i,
                               cv::Mat &res_color3f,cv::Mat &res_color_num,
                               double ratio = 0.95, const int clrNums[3] = DefaultNums);

 private:
  static const int DefaultNums[3];

  class Region {
   public:
    Region() { pix_num = 0; ad2c = Point2d(0, 0);}
    int pix_num;  // Number of pixels
    vector<CostfIdx> fre_idx;  // Frequency of each color and its index
    Point2d centroid;
    Point2d ad2c; // Average distance to image center
  };

  static void BuildRegions(const cv::Mat &region_idx1i, vector<Region> &regs,
                           const cv::Mat &color_idx1i, int color_num);

  static void RegionContrastCore(const vector<Region> &regs,
                                 const cv::Mat& color3fv, cv::Mat& reg_sal1dv,
                                 double sigma_dist);

  // Get border regions, which typically corresponds to background region
  static Mat GetBorderReg(const cv::Mat &idx1i, int reg_num, double ratio = 0.02,
                          double thr = 0.3);

  static void SmoothSaliency(cv::Mat& sal1f, float delta,
                             const vector<vector<CostfIdx>>& similar);
  static void SmoothSaliency(const cv::Mat& colorNum1i, cv::Mat& sal1f,
                             float delta,
                             const vector<vector<CostfIdx>>& similar);

};

}



#endif
