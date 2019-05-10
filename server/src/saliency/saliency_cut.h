#ifndef SERVER_SALIENCY_SALIENCY_CUT_H_
#define SERVER_SALIENCY_SALIENCY_CUT_H_

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/core/matx.hpp>

#include <basic/graph.h>
#include <cluster/gaussian_mixture_models.h>

namespace saliencycut {

class SaliencyCut {
 public: // Functions for saliency cut
  SaliencyCut(const cv::Mat& img3f);
  ~SaliencyCut(void);
  // User supplied Trimap values
  enum TrimapValue {TrimapBackground = 0, TrimapUnknown = 128, TrimapForeground = 255};

 public:


  static int ProcessSingleImg(const std::string& img_path,
                              std::string& result_rc_path,
                              std::string& result_rcc_path);
  static int ProcessImages(const std::string& root_dir_path, int& amount, int& time_cost);

  static void ShowImageInfo(const cv::Mat& img);

  // Refer initialize for parameters
  static cv::Mat CutObjs(const cv::Mat& img3f, const cv::Mat& sal1f,
                         float t1 = 0.2f, float t2 = 0.9f,
                         const cv::Mat& borderMask = cv::Mat(), int wkSize = 20);





 public: // Functions for GrabCut

  // Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper
  void initialize(const cv::Rect& rect);

  // Initialize using saliency map. In the Trimap: background < t1, foreground > t2, others unknown.
  // Saliency values are in [0, 1], "sal1f" and "1-sal1f" are used as weight to train fore and back ground GMMs
  void initialize(const cv::Mat& sal1f, float t1, float t2);
  void initialize(const cv::Mat& sal1u); // Background = 0, unknown = 128, foreground = 255

  void fitGMMs();
  // Run Grabcut refinement on the hard segmentation
  void refine() {int changed = 1; while (changed) changed = refineOnce();}
  int refineOnce();

  // Draw result
  void drawResult(cv::Mat& maskForeGround) {
    compare(_segVal1f, 0.5, maskForeGround, CMP_GT);
  }


 private:
  // Update hard segmentation after running GraphCut,
  // Returns the number of pixels that have changed from foreground to background or vice versa.
  int updateHardSegmentation();
  void initGraph();	// builds the graph for GraphCut
  // Return number of difference and then expand fMask to get mask1u.
  static int ExpandMask(const cv::Mat& fMask, cv::Mat& mask1u,
                        const cv::Mat& bdReg1u, int expandRatio = 5);





 private:
  int _w, _h;		// Width and height of the source image
  cv::Mat _imgBGR3f, _imgLab3f; // BGR images is used to find GMMs and Lab for pixel distance
  cv::Mat _trimap1i;	// Trimap value
  cv::Mat _segVal1f;	// Hard segmentation with type SegmentationValue

  // Variables used in formulas from the paper.
  float _lambda;		// lambda = 50. This value was suggested the GrabCut paper.
  float _beta;		// beta = 1 / ( 2 * average of the squared color distances between all pairs of neighboring pixels (8-neighborhood) )
  float _L;			// L = a large value to force a pixel to be foreground or background
  GraphF *_graph;

  // Storage for N-link weights, each pixel stores links to only four of its 8-neighborhood neighbors.
  // This avoids duplication of links, while still allowing for relatively easy lookup.
  // First 4 directions in DIRECTION8 are: right, rightBottom, bottom, leftBottom.
  cv::Mat_<Vec4f> _NLinks;

  int _directions[4]; // From DIRECTION8 for easy location

  CmGMM _bGMM, _fGMM; // Background and foreground GMM
  cv::Mat _bGMMidx1i, _fGMMidx1i;	// Background and foreground GMM components, supply memory for GMM, not used for Grabcut
  cv::Mat _show3u; // Image for display medial results



};
}

#endif  // SERVER_SALIENCY_SALIENCY_CUT_H_
