#include "saliency_cut.h"

#include <iostream>
#include <ctime>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

#include <saliency/saliency_region_contrast.h>
#include <cluster/gaussian_mixture_models.h>
#include <saliency/some_definition.h>
#include <basic/image_operations.h>
#include <basic/ytplatform.h>

using namespace std;
using namespace cv;

namespace saliencycut {
SaliencyCut::SaliencyCut(const Mat& img3f)
    :_fGMM(5), _bGMM(5), _w(img3f.cols), _h(img3f.rows), _lambda(50) {
  CV_Assert(img3f.data != NULL && img3f.type() == CV_32FC3);
  _imgBGR3f = img3f;
  cvtColor(_imgBGR3f, _imgLab3f, cv::COLOR_BGR2Lab);
  _trimap1i = Mat::zeros(_h, _w, CV_32S);
  _segVal1f = Mat::zeros(_h, _w, CV_32F);
  _graph = NULL;

  _L = 8 * _lambda + 1;// Compute L
  _beta = 0; {// compute beta
    int edges = 0;
    double result = 0;
    for (int y = 0; y < _h; ++y) {
      const Vec3f* img = _imgLab3f.ptr<Vec3f>(y);
      for (int x = 0; x < _w; ++x){
        Point pnt(x, y);
        for (int i = 0; i < 4; i++)	{
          Point pntN = pnt + DIRECTION8[i];
          if (CHK_IND(pntN))
            result += vecSqrDist(_imgLab3f.at<Vec3f>(pntN), img[x]), edges++;
        }
      }
    }
    _beta = (float)(0.5 * edges/result);
  }
  _NLinks.create(_h, _w); {// computeNLinks
    static const float dW[4] = {1, (float)(1/SQRT2), 1, (float)(1/SQRT2)};
    for (int y = 0; y < _h; y++) {
      Vec4f *nLink = _NLinks.ptr<Vec4f>(y);
      const Vec3f* img = _imgLab3f.ptr<Vec3f>(y);
      for (int x = 0; x < _w; x++, nLink++) {
        Point pnt(x, y);
        const Vec3f& c1 = img[x];
        for (int i = 0; i < 4; i++)	{
          Point pntN = pnt + DIRECTION8[i];
          if (CHK_IND(pntN))
            (*nLink)[i] = _lambda * dW[i] * exp(-_beta * vecSqrDist(_imgLab3f.at<Vec3f>(pntN), c1));
        }
      }
    }
  }

  for (int i = 0; i < 4; i++)
    _directions[i] = DIRECTION8[i].x + DIRECTION8[i].y * _w;
}

SaliencyCut::~SaliencyCut(void) {
  if (_graph)
    delete _graph;
}


int SaliencyCut::ProcessSingleImg(const string& img_path,
                                  string& result_rc_path,
                                  string& result_rcc_path) {
  cout << "cpp: " << __cplusplus << endl;
  int end_pos = img_path.rfind(".");
  string str_name = img_path.substr(0, end_pos) + "_" + to_string(std::time(0));
  result_rc_path = str_name + "_RC.png"; // Region contrast
  result_rcc_path = str_name + "_RCC.png"; // Region contrast cut
  //first error detection, permission and presense of the image
  //imread 2nd parameter 0: grayscale; 1: 3 channels; -1: as is(with alpha channel)
  Mat img3f = imread(img_path, 1); // 3u now
  if (!img3f.data) {
    cout << "empty image" << endl;
    return -1;
  }
  //convert to float, 3 channels
  img3f.convertTo(img3f, CV_32FC3, 1.0/255, 0);


  Mat sal = regioncontrast::RegionContrast::GetRegionContrast(img3f);
  // imshow("sal", sal);
  // waitKey(0);
  // cout << "M" << endl << endl << endl << sal << endl;
  // save region contrast image
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  imwrite(result_rc_path, sal*255, compression_params);
  // cout << "finish stage 1" << endl;

  Mat cutMat;
  float t = 0.9f;
  int maxIt = 4;
  GaussianBlur(sal, sal, Size(9, 9), 0);
  normalize(sal, sal, 0, 1, NORM_MINMAX);
  while (cutMat.empty() && maxIt--){
    cutMat = saliencycut::SaliencyCut::CutObjs(img3f, sal, 0.1f, t);
    t -= 0.2f;
  }
  if (!cutMat.empty()) {
    imwrite(result_rcc_path, cutMat);
    // cout << "M" << endl << cutMat << endl;
  }
  else
    cout << "EEEOR! While saving rcc" << endl;

  // finish region based saliency region detection

  return 0;
}


int SaliencyCut::ProcessImages(const std::string& root_dir_path, int& amount, int& time_cost) {

  time_t start_time = std::time(0);
  int image_amount = -1;
  vector<string> image_names;
  image_names.reserve(100);   // reserve 100 file for the moment
  ytfile::get_file_names(root_dir_path, image_names);
  image_amount = image_names.size();

  // Then create folder, so the file_names won't contain result folder.
  string saliency_dir_path = root_dir_path+ "/" + "cut_result_" + to_string(std::time(0));
  if (ytfile::file_exist(saliency_dir_path)) {
    if (!ytfile::is_dir(saliency_dir_path)) {
      cout << "Please rename your file 'cut_result'" << endl;
      return -1;
    }
  }
  else {
    ytfile::mk_dir(saliency_dir_path);
  }
  // Finish make directory; Start cutting

#pragma omp parallel for
  for (int i = 0; i < image_amount; ++i){
    string img_path = image_names[i];
    int end_pos = img_path.rfind(".");
    string str_name = img_path.substr(0, end_pos) + "_" + to_string(std::time(0));
    string result_rc_path = str_name + "_RC.png"; // Region contrast
    string result_rcc_path = str_name + "_RCC.png"; // Region contrast cut
    printf("OpenMP Test, thread index: %d\n", omp_get_thread_num());

    Mat img3f = imread(root_dir_path+ "/" + img_path);
    CV_Assert_(img3f.data != NULL, ("Can't load image \n"));
    img3f.convertTo(img3f, CV_32FC3, 1.0/255);
    Mat sal = regioncontrast::RegionContrast::GetRegionContrast(img3f);
    //imwrite(saliency_dir_path + "/" + result_rc_path, sal*255);
    // Finish First Stage

    Mat cutMat;
    float t = 0.9f;
    int maxIt = 4;
    GaussianBlur(sal, sal, Size(9, 9), 0);
    normalize(sal, sal, 0, 1, NORM_MINMAX);
    while (cutMat.empty() && maxIt--){
      cutMat = saliencycut::SaliencyCut::CutObjs(img3f, sal, 0.1f, t);
      t -= 0.2f;
    }
    if (!cutMat.empty())
      imwrite(saliency_dir_path + "/" + result_rcc_path, cutMat);
    else
      cout << "EEEOR! While saving rcc" << endl;
  }
  amount = image_amount;
  time_cost = (int) std::time(0) - start_time;
  return image_amount;
}


void SaliencyCut::ShowImageInfo(const Mat& img) {
  // +--------+----+----+----+----+------+------+------+------+
  // |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
  // +--------+----+----+----+----+------+------+------+------+
  // | CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
  // | CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
  // | CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
  // | CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
  // | CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
  // | CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
  // | CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
  // +--------+----+----+----+----+------+------+------+------+

  cout << "M = "<< endl << " "  << img << endl << endl;

  std::cout << " channels " << img.channels()
            << " type " << img.type()
            << " size " << img.size()
            << std::endl;

  // cv::Vec3b vec3b = img.at<cv::Vec3b>(0,0);
  // uchar vec3b0 = img.at<cv::Vec3b>(0,0)[0];
  // uchar vec3b1 = img.at<cv::Vec3b>(0,0)[1];
  // uchar vec3b2 = img.at<cv::Vec3b>(0,0)[2];
  // std::cout<<"vec3b = "<<vec3b<<std::endl;
  // std::cout<<"vec3b0 = "<<(int)vec3b0<<std::endl;
  // std::cout<<"vec3b1 = "<<(int)vec3b1<<std::endl;
  // std::cout<<"vec3b2 = "<<(int)vec3b2<<std::endl;

  cv::Vec3d vec3d = img.at<cv::Vec3d>(0,0);
  double vec3d0 = img.at<cv::Vec3d>(0,0)[0];
  double vec3d1 = img.at<cv::Vec3d>(0,0)[1];
  double vec3d2 = img.at<cv::Vec3d>(0,0)[2];
  std::cout<<"vec3d = "<<vec3d<<std::endl;

}



Mat SaliencyCut::CutObjs(const Mat& _img3f, const Mat& _sal1f,
                         float t1, float t2, const Mat& _border1u,
                         int wkSize) {
  Mat border1u = _border1u;
  if (border1u.data == NULL || border1u.size != _img3f.size){
    int bW = cvRound(0.02 * _img3f.cols), bH = cvRound(0.02 * _img3f.rows);
    border1u.create(_img3f.rows, _img3f.cols, CV_8U);
    border1u = 255;
    border1u(Rect(bW, bH, _img3f.cols - 2*bW, _img3f.rows - 2*bH)) = 0;
  }
  Mat sal1f, wkMask;
  _sal1f.copyTo(sal1f);
  sal1f.setTo(0, border1u);

  cv::Rect rect(0, 0, _img3f.cols, _img3f.rows);
  if (wkSize > 0){
    threshold(sal1f, sal1f, t1, 1, THRESH_TOZERO);
    sal1f.convertTo(wkMask, CV_8U, 255);
    threshold(wkMask, wkMask, 70, 255, THRESH_TOZERO);
    wkMask = imageoperations::ImageOperations::GetLargestSumNoneZeroRegion(wkMask, 0.005);
    if (wkMask.data == NULL)
      return Mat();
    rect = imageoperations::ImageOperations::GetMaskRange(wkMask, wkSize);
    sal1f = sal1f(rect);
    border1u = border1u(rect);
    wkMask = wkMask(rect);
  }
  const Mat img3f = _img3f(rect);

  Mat fMask;
  SaliencyCut salCut(img3f);
  salCut.initialize(sal1f, t1, t2);
  const int outerIter = 4;
  //salCut.showMedialResults("Ini");
  for (int j = 0; j < outerIter; j++)	{
    salCut.fitGMMs();
    int changed = 1000, times = 8;
    while (changed > 50 && times--) {
      changed = salCut.refineOnce();
    }
    salCut.drawResult(fMask);

    fMask = imageoperations::ImageOperations::GetLargestSumNoneZeroRegion(fMask);
    if (fMask.data == NULL)
      return Mat();

    if (j == outerIter - 1 || ExpandMask(fMask, wkMask, border1u, 5) < 10)
      break;

    salCut.initialize(wkMask);
    fMask.copyTo(wkMask);
  }

  Mat resMask = Mat::zeros(_img3f.size(), CV_8U);
  fMask.copyTo(resMask(rect));
  return resMask;
}


// Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper
void SaliencyCut::initialize(const Rect &rect) {
  _trimap1i = TrimapBackground;
  _trimap1i(rect) = TrimapUnknown;
  _segVal1f = 0;
  _segVal1f(rect) = 1;
}

// Background = 0, unknown = 128, foreground = 255
void SaliencyCut::initialize(const Mat& sal1u) {
  CV_Assert(sal1u.type() == CV_8UC1 && sal1u.size == _imgBGR3f.size);
  for (int y = 0; y < _h; y++) {
    int* triVal = _trimap1i.ptr<int>(y);
    const byte *salVal = sal1u.ptr<byte>(y);
    float *segVal = _segVal1f.ptr<float>(y);
    for (int x = 0; x < _w; x++) {
      triVal[x] = salVal[x] < 70 ? TrimapBackground : TrimapUnknown;
      triVal[x] = salVal[x] > 200 ? TrimapForeground : triVal[x];
      segVal[x] = salVal[x] < 70 ? 0 : 1.f;
    }
  }
}

// Initialize using saliency map. In the Trimap: background < t1, foreground > t2, others unknown.
// Saliency values are in [0, 1], "sal1f" and "1-sal1f" are used as weight to train fore and back ground GMMs
void SaliencyCut::initialize(const cv::Mat& sal1f,
                             float t1, float t2) {
  CV_Assert(sal1f.type() == CV_32F && sal1f.size == _imgBGR3f.size);
  sal1f.copyTo(_segVal1f);

  for (int y = 0; y < _h; y++) {
    int* triVal = _trimap1i.ptr<int>(y);
    const float *segVal = _segVal1f.ptr<float>(y);
    for (int x = 0; x < _w; x++) {
      triVal[x] = segVal[x] < t1 ? TrimapBackground : TrimapUnknown;
      triVal[x] = segVal[x] > t2 ? TrimapForeground : triVal[x];
    }
  }
}

void SaliencyCut::fitGMMs() {
  _fGMM.BuildGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);
  _bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, 1 - _segVal1f);
}

int SaliencyCut::refineOnce() {
  // Steps 4 and 5: Learn new GMMs from current segmentation
  if (_fGMM.GetSumWeight() < 50 || _bGMM.GetSumWeight() < 50)
    return 0;

  _fGMM.RefineGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);
  _bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, 1 - _segVal1f);

  // Step 6: Run GraphCut and update segmentation
  initGraph();
  if (_graph)
    _graph->maxflow();

  return updateHardSegmentation();
}

void SaliencyCut::initGraph() {
  // Set up the graph (it can only be used once, so we have to recreate it each time the graph is updated)
  if (_graph == NULL)
    _graph = new GraphF(_w * _h, 4 * _w * _h);
  else
    _graph->reset();
  _graph->add_node(_w * _h);

  for (int y = 0, id = 0; y < _h; ++y) {
    int* triMapD = _trimap1i.ptr<int>(y);
    const float* img = _imgBGR3f.ptr<float>(y);
    for(int x = 0; x < _w; x++, img += 3, id++) {
      float back, fore;
      if (triMapD[x] == TrimapUnknown ) {
        fore = -log(_bGMM.P(img));
        back = -log(_fGMM.P(img));
      }
      else if (triMapD[x] == TrimapBackground )
        fore = 0, back = _L;
      else		// TrimapForeground
        fore = _L,	back = 0;

      // Set T-Link weights
      _graph->add_tweights(id, fore, back); // _graph->set_tweights(_nodes(y, x), fore, back);

      // Set N-Link weights from precomputed values
      Point pnt(x, y);
      const Vec4f& nLink = _NLinks(pnt);
      for (int i = 0; i < 4; i++)	{
        Point nPnt = pnt + DIRECTION8[i];
        if (CHK_IND(nPnt))
          _graph->add_edge(id, id + _directions[i], nLink[i], nLink[i]);
      }
    }
  }
}


int SaliencyCut::ExpandMask(const cv::Mat& fMask, cv::Mat& mask1u,
                            const cv::Mat& bdReg1u, int expandRatio) {
  compare(fMask, mask1u, mask1u, CMP_NE);
  int changed = cvRound(sum(mask1u).val[0] / 255.0);

  Mat bigM, smalM;
  dilate(fMask, bigM, Mat(), Point(-1, -1), expandRatio);
  erode(fMask, smalM, Mat(), Point(-1, -1), expandRatio);
  static const double erodeSmall = 255 * 50;
  if (sum(smalM).val[0] < erodeSmall)
    smalM = fMask;
  mask1u = bigM * 0.5 + smalM * 0.5;
  mask1u.setTo(0, bdReg1u);
  return changed;
}


int SaliencyCut::updateHardSegmentation() {
  int changed = 0;
  for (int y = 0, id = 0; y < _h; ++y) {
    float* segVal = _segVal1f.ptr<float>(y);
    int* triMapD = _trimap1i.ptr<int>(y);
    for (int x = 0; x < _w; ++x, id++) {
      float oldValue = segVal[x];
      if (triMapD[x] == TrimapBackground)
        segVal[x] = 0.f; // SegmentationBackground
      else if (triMapD[x] == TrimapForeground)
        segVal[x] = 1.f; // SegmentationForeground
      else
        segVal[x] = _graph->what_segment(id) == GraphF::SOURCE ? 1.f : 0.f;
      changed += abs(segVal[x] - oldValue) > 0.1 ? 1 : 0;
    }
  }
  return changed;
}

} // end namespace saliencycut
