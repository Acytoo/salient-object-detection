#include <saliency/saliency_region_contrast.h>
#include <saliency/saliency_cut.h>
#include <basic/block.h>
#include <basic/graph.h>
#include <basic/segment_image.h>

#include <iostream>
#include <map>

#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

namespace regioncontrast {

  const int RegionContrast::DefaultNums[3] = {12, 12, 12};

  Mat RegionContrast::GetRegionContrast(const cv::Mat &img3f,
                                        double sigma_dist,
                                        double seg_k,
                                        int seg_min_size,
                                        double seg_sigma) {
    Mat img_lab3f, region_index1i;
    //cvtColor BGR2Lab currently not support 16bit img
    //see at https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_lab Alec@2019.3.8
    cvtColor(img3f, img_lab3f, CV_BGR2Lab);
    int region_num = SegmentImage(img_lab3f, region_index1i,
                                  seg_sigma, seg_k, seg_min_size);
    return GetRegionContrast(img3f, region_index1i, region_num, sigma_dist);
  }


  Mat RegionContrast::GetRegionContrast(const cv::Mat& img3f,
                                        const cv::Mat& region_index1i,
                                        int reg_num,
                                        double sigma_dist) {
    Mat color_idx1i, reg_sal1v, tmp, color3fv;
    int quantize_num = Quantize(img3f, color_idx1i, color3fv, tmp);  // Color quantization
    // cout << "quantize_num " << quantize_num << endl;
    if (quantize_num == 2){
      Mat sal;
      compare(color_idx1i, 1, sal, CMP_EQ);
      sal.convertTo(sal, CV_32F, 1.0/255);
      return sal;
    }
    if (quantize_num <= 1)
      return Mat::zeros(img3f.size(), CV_32F);

    cvtColor(color3fv, color3fv, CV_BGR2Lab);
    vector<Region> regs(reg_num);
    BuildRegions(region_index1i, regs, color_idx1i, color3fv.cols);
    RegionContrastCore(regs, color3fv, reg_sal1v, sigma_dist);

    Mat sal1f = Mat::zeros(img3f.size(), CV_32F);
    cv::normalize(reg_sal1v, reg_sal1v, 0, 1, NORM_MINMAX, CV_32F);
    float* reg_sal = (float*)reg_sal1v.data;
    for (int r = 0; r < img3f.rows; r++) {
      const int* p_reg_idx = region_index1i.ptr<int>(r);
      float* sal = sal1f.ptr<float>(r);
      for (int c = 0; c < img3f.cols; c++)
        sal[c] = reg_sal[p_reg_idx[c]];
    }

    Mat bdReg1u = GetBorderReg(region_index1i, reg_num, 0.02, 0.4);
    sal1f.setTo(0, bdReg1u);
    SmoothByHist(img3f, sal1f, 0.1f);
    SmoothByRegion(sal1f, region_index1i, reg_num);
    sal1f.setTo(0, bdReg1u);

    GaussianBlur(sal1f, sal1f, Size(3, 3), 0);
    return sal1f;

  }


  int RegionContrast::Quantize(const cv::Mat &img3f,
                               cv::Mat &color_idx1i,
                               cv::Mat &res_color3f,
                               cv::Mat &res_color_num,
                               double ratio,
                               const int clrNums[3]) {
    float clrTmp[3] = {clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f}; // 11.9999, 11.9999, 11.9999
    int w[3] = {clrNums[1] * clrNums[2], clrNums[2], 1}; // 144, 12, 1
    CV_Assert(img3f.data != NULL); // works in opencv 4
    color_idx1i = Mat::zeros(img3f.size(), CV_32S);
    int rows = img3f.rows, cols = img3f.cols;
    if (img3f.isContinuous() && color_idx1i.isContinuous()) { // Called in 2nd times: might not continus, accelerate
      cols *= rows;
      rows = 1;
    }

    // Build color pallet{color_identifier:occured_times}
    map<int, int> pallet;
    for (int y = 0; y < rows; ++y) {
      const float* p_ori_img = img3f.ptr<float>(y);
      int* p_color_idx = color_idx1i.ptr<int>(y);
      for (int x = 0; x < cols; ++x, p_ori_img += 3) { // (B*144 + G*12 + r*1) * 11.9999 for 1st quantize
        p_color_idx[x] = (int)(p_ori_img[0]*clrTmp[0])*w[0] +
          (int)(p_ori_img[1]*clrTmp[1])*w[1] +
          (int)(p_ori_img[2]*clrTmp[2]);
        pallet[p_color_idx[x]]++;
      }
    }

    // Find significant colors
    int max_num = 0;
    int count = 0;
    // Sort the pallet by its second value, we need to use vector<pair<>>, since map(RBTree) is sorted by its first value
    vector<pair<int, int>> num; // num{times, id}
    num.reserve(pallet.size());
    for (map<int, int>::iterator it = pallet.begin(), stop = pallet.end(); it != stop; ++it)
      num.push_back(pair<int, int>(it->second, it->first)); // Second: color occured frequency; first: color identifier
    sort(num.begin(), num.end(), std::greater<pair<int, int>>()); // sort default: sort(vect.begin(), vect.end(), less<int>());

    max_num = (int)num.size(); // max_num = num.size();
    int max_drop_num = cvRound(rows * cols * (1-ratio));
    for (int current_freq = num[max_num-1].first; current_freq < max_drop_num && 1 < max_num; --max_num)
      current_freq += num[max_num - 2].first;
    max_num = min(max_num, 256); // To avoid very rarely case
    if (max_num <= 10)
      max_num = min(10, (int)num.size());

    pallet.clear();
    for (int i = 0; i < max_num; ++i)
      pallet[num[i].second] = i;
    // pallet{id:color_precedence}, size = max_num

    // high frequency color
    vector<Vec3i> color3i(num.size());
    for (unsigned int i = 0, stop = num.size(); i < stop; ++i) {
      color3i[i][0] = num[i].second / w[0];
      color3i[i][1] = num[i].second % w[0] / w[1];
      color3i[i][2] = num[i].second % w[1];
    }
    // low frequency color: find the nearest high-freq color and replace it
    for (unsigned int i = max_num, stop = num.size(); i < stop; ++i) {
      int sim_idx = 0, sim_val = INT_MAX;
      for (int j = 0; j < max_num; ++j) {
        int d_ij = vecSqrDist<int, 3>(color3i[i], color3i[j]);
        if (d_ij < sim_val)
          sim_val = d_ij, sim_idx = j;
      }
      pallet[num[i].second] = pallet[num[sim_idx].second]; // pallet size: all color size
    }

    // max_num: left high frequency color's number
    res_color3f = Mat::zeros(1, max_num, CV_32FC3); // color Mat(vector) of high-freq color
    res_color_num = Mat::zeros(res_color3f.size(), CV_32S); // the number of occurences of each high-freq color

    Vec3f* p_color = (Vec3f*)(res_color3f.data);
    int* p_color_num = (int*)(res_color_num.data);
    // For each pixel
    for (int y = 0; y < rows; ++y) {
      const Vec3f* p_ori_img = img3f.ptr<Vec3f>(y);
      int* p_color_idx = color_idx1i.ptr<int>(y);
      for (int x = 0; x < cols; ++x) {
        p_color_idx[x] = pallet[p_color_idx[x]]; // 1i
        p_color[p_color_idx[x]] += p_ori_img[x]; // 3f
        p_color_num[p_color_idx[x]] ++;
      }
    }
    for (int i = 0; i < res_color3f.cols; ++i)
      p_color[i] /= (float)p_color_num[i];

    return res_color3f.cols; // the number of high-freq color
  }


  void RegionContrast::BuildRegions(const cv::Mat &region_idx1i,
                                    vector<Region> &regs,
                                    const cv::Mat &color_idx1i,
                                    int color_num) {
    int rows = region_idx1i.rows, cols = region_idx1i.cols, reg_num = (int)regs.size();
    double cx = cols/2.0, cy = rows / 2.0;
    Mat_<int> reg_color_freq1i  = Mat_<int>::zeros(reg_num, color_num); // region color frequency
    for (int y = 0; y < rows; ++y) {
      const int *p_reg_idx = region_idx1i.ptr<int>(y);
      const int *p_color_idx = color_idx1i.ptr<int>(y);
      for (int x = 0; x < cols; ++x, ++p_reg_idx, ++p_color_idx) {
        Region &reg = regs[*p_reg_idx];
        reg.pix_num++;
        reg.centroid.x += x;
        reg.centroid.y += y;
        reg_color_freq1i(*p_reg_idx, *p_color_idx)++;
        reg.ad2c += Point2d(abs(x - cx), abs(y - cy));
      }
    }

    for (int i = 0; i < reg_num; ++i){
      Region &reg = regs[i];
      reg.centroid.x /= reg.pix_num * cols;
      reg.centroid.y /= reg.pix_num * rows;
      reg.ad2c.x /= reg.pix_num * cols;
      reg.ad2c.y /= reg.pix_num * rows;
      int *p_reg_color_freq = reg_color_freq1i.ptr<int>(i);
      for (int j = 0; j < color_num; ++j){
        float fre = (float)p_reg_color_freq[j]/(float)reg.pix_num;
        if (p_reg_color_freq[j] > EPS)
          reg.fre_idx.push_back(make_pair(fre, j));
      }
    }
  }


  void RegionContrast::RegionContrastCore(const vector<Region> &regs,
                                          const cv::Mat &color3fv,
                                          cv::Mat &regSal1d,
                                          double sigmaDist) {
    Mat_<float> cDistCache1f = Mat::zeros(color3fv.cols, color3fv.cols, CV_32F);
    {
      Vec3f* pColor = (Vec3f*)color3fv.data;
      for(int i = 0; i < cDistCache1f.rows; i++)
        for(int j= i+1; j < cDistCache1f.cols; j++)
          cDistCache1f[i][j] = cDistCache1f[j][i] = vecDist<float, 3>(pColor[i], pColor[j]);
    }

    int regNum = (int)regs.size();
    Mat_<double> rDistCache1d = Mat::zeros(regNum, regNum, CV_64F);
    regSal1d = Mat::zeros(1, regNum, CV_64F);
    double* regSal = (double*)regSal1d.data;
    for (int i = 0; i < regNum; i++){
      const Point2d &rc = regs[i].centroid;
      for (int j = 0; j < regNum; j++){
        if(i<j) {
          double dd = 0;
          const vector<CostfIdx> &c1 = regs[i].fre_idx, &c2 = regs[j].fre_idx;
          for (size_t m = 0; m < c1.size(); m++)
            for (size_t n = 0; n < c2.size(); n++)
              dd += cDistCache1f[c1[m].second][c2[n].second] * c1[m].first * c2[n].first;
          rDistCache1d[j][i] = rDistCache1d[i][j] = dd * exp(-pntSqrDist(rc, regs[j].centroid)/sigmaDist);
        }
        regSal[i] += regs[j].pix_num * rDistCache1d[i][j];
      }
      regSal[i] *= exp(-9.0 * (sqr(regs[i].ad2c.x) + sqr(regs[i].ad2c.y)));
    }

  }


  Mat RegionContrast::GetBorderReg(const cv::Mat &idx1i,
                                   int regNum, double ratio,
                                   double thr) {
    // Variance of x and y
    vector<double> vX(regNum), vY(regNum);
    int w = idx1i.cols, h = idx1i.rows;{
      vector<double> mX(regNum), mY(regNum), n(regNum); // Mean value of x and y, pixel number of region
      for (int y = 0; y < idx1i.rows; ++y){
        const int *idx = idx1i.ptr<int>(y);
        for (int x = 0; x < idx1i.cols; x++, ++idx)
          mX[*idx] += x, mY[*idx] += y, n[*idx]++;
      }
      for (int i = 0; i < regNum; i++)
        mX[i] /= n[i], mY[i] /= n[i];
      for (int y = 0; y < idx1i.rows; ++y){
        const int *idx = idx1i.ptr<int>(y);
        for (int x = 0; x < idx1i.cols; x++, ++idx)
          vX[*idx] += abs(x - mX[*idx]), vY[*idx] += abs(y - mY[*idx]);
      }
      for (int i = 0; i < regNum; i++)
        vX[i] = vX[i]/n[i] + EPS, vY[i] = vY[i]/n[i] + EPS;
    }

    // Number of border pixels in x and y border region
    vector<int> xbNum(regNum), ybNum(regNum);
    int wGap = cvRound(w * ratio), hGap = cvRound(h * ratio);
    vector<Point> bPnts; {
      ForPoints2(pnt, 0, 0, w, hGap) // Top region
        ybNum[idx1i.at<int>(pnt)]++, bPnts.push_back(pnt);
      ForPoints2(pnt, 0, h - hGap, w, h) // Bottom region
        ybNum[idx1i.at<int>(pnt)]++, bPnts.push_back(pnt);
      ForPoints2(pnt, 0, 0, wGap, h) // Left region
        xbNum[idx1i.at<int>(pnt)]++, bPnts.push_back(pnt);
      ForPoints2(pnt, w - wGap, 0, w, h)
        xbNum[idx1i.at<int>(pnt)]++, bPnts.push_back(pnt);
    }

    Mat bReg1u(idx1i.size(), CV_8U);{  // likelihood map of border region
      double xR = 1.0/(4*hGap), yR = 1.0/(4*wGap);
      vector<byte> regL(regNum); // likelihood of each region belongs to border background
      for (int i = 0; i < regNum; i++) {
        double lk = xbNum[i] * xR / vY[i] + ybNum[i] * yR / vX[i];
        regL[i] = lk/thr > 1 ? 255 : 0; //saturate_cast<byte>(255 * lk / thr);
      }

      for (int r = 0; r < h; ++r)	{
        const int *idx = idx1i.ptr<int>(r);
        byte* maskData = bReg1u.ptr<byte>(r);
        for (int c = 0; c < w; ++c, ++idx)
          maskData[c] = regL[*idx];
      }
    }

    for (size_t i = 0; i < bPnts.size(); i++)
      bReg1u.at<byte>(bPnts[i]) = 255;
    return bReg1u;

  }


  void RegionContrast::SmoothByHist(const cv::Mat &img3f,
                                    cv::Mat &sal1f, float delta) {
    //imshow("Before", sal1f); imshow("Src", img3f);

    // Quantize colors
    CV_Assert(img3f.size() == sal1f.size() && img3f.type() == CV_32FC3 && sal1f.type() == CV_32FC1);
    Mat idx1i, binColor3f, colorNums1i;
    int binN = Quantize(img3f, idx1i, binColor3f, colorNums1i);

    // Get initial color saliency
    Mat _colorSal =  Mat::zeros(1, binN, CV_64FC1);
    int rows = img3f.rows, cols = img3f.cols;{
      double* colorSal = (double*)_colorSal.data;
      if (img3f.isContinuous() && sal1f.isContinuous())
        cols *= img3f.rows, rows = 1;
      for (int y = 0; y < rows; ++y){
        const int* idx = idx1i.ptr<int>(y);
        const float* initialS = sal1f.ptr<float>(y);
        for (int x = 0; x < cols; x++)
          colorSal[idx[x]] += initialS[x];
      }
      const int *colorNum = (int*)(colorNums1i.data);
      for (int i = 0; i < binN; i++)
        colorSal[i] /= colorNum[i];
      normalize(_colorSal, _colorSal, 0, 1, NORM_MINMAX, CV_32F);
    }
    // Find similar colors & Smooth saliency value for color bins
    vector<vector<CostfIdx>> similar(binN); // Similar color: how similar and their index
    Vec3f* color = (Vec3f*)(binColor3f.data);
    cvtColor(binColor3f, binColor3f, CV_BGR2Lab);
    for (int i = 0; i < binN; i++){
      vector<CostfIdx> &similari = similar[i];
      similari.push_back(make_pair(0.f, i));
      for (int j = 0; j < binN; j++)
        if (i != j)
          similari.push_back(make_pair(vecDist<float, 3>(color[i], color[j]), j));
      sort(similari.begin(), similari.end());
    }
    cvtColor(binColor3f, binColor3f, CV_Lab2BGR);
    //CmShow::HistBins(binColor3f, _colorSal, "BeforeSmooth", true);
    SmoothSaliency(colorNums1i, _colorSal, delta, similar);
    //CmShow::HistBins(binColor3f, _colorSal, "AfterSmooth", true);

    // Reassign pixel saliency values
    float* colorSal = (float*)(_colorSal.data);
    for (int y = 0; y < rows; ++y){
      const int* idx = idx1i.ptr<int>(y);
      float* resSal = sal1f.ptr<float>(y);
      for (int x = 0; x < cols; x++)
        resSal[x] = colorSal[idx[x]];
    }
    //imshow("After", sal1f);
    //waitKey(0);
  }

  void RegionContrast::SmoothByRegion(cv::Mat &sal1f,
                                      const cv::Mat &segIdx1i,
                                      int regNum,
                                      bool bNormalize) {
    vector<double> saliecy(regNum, 0);
    vector<int> counter(regNum, 0);
    for (int y = 0; y < sal1f.rows; ++y){
      const int *idx = segIdx1i.ptr<int>(y);
      float *sal = sal1f.ptr<float>(y);
      for (int x = 0; x < sal1f.cols; x++){
        saliecy[idx[x]] += sal[x];
        counter[idx[x]] ++;
      }
    }

    for (size_t i = 0; i < counter.size(); i++)
      saliecy[i] /= counter[i];
    Mat rSal(1, regNum, CV_64FC1, &saliecy[0]);
    if (bNormalize)
      normalize(rSal, rSal, 0, 1, NORM_MINMAX);

    for (int y = 0; y < sal1f.rows; ++y){
      const int *idx = segIdx1i.ptr<int>(y);
      float *sal = sal1f.ptr<float>(y);
      for (int x = 0; x < sal1f.cols; x++)
        sal[x] = (float)saliecy[idx[x]];
    }

  }


  void RegionContrast::SmoothSaliency(cv::Mat &sal1f,
                                      float delta,
                                      const vector<vector<CostfIdx> > &similar) {
    Mat colorNum1i = Mat::ones(sal1f.size(), CV_32SC1);
    SmoothSaliency(colorNum1i, sal1f, delta, similar);
  }

  void RegionContrast::SmoothSaliency(const cv::Mat &colorNum1i,
                                      cv::Mat &sal1f,
                                      float delta,
                                      const vector<vector<CostfIdx> > &similar) {
    if (sal1f.cols < 2)
      return;
    CV_Assert(sal1f.rows == 1 && sal1f.type() == CV_32FC1);
    CV_Assert(colorNum1i.size() == sal1f.size() && colorNum1i.type() == CV_32SC1);

    int binN = sal1f.cols;
    Mat newSal1d= Mat::zeros(1, binN, CV_64FC1);
    float *sal = (float*)(sal1f.data);
    double *newSal = (double*)(newSal1d.data);
    int *pW = (int*)(colorNum1i.data);

    // Distance based smooth
    int n = max(cvRound(binN * delta), 2);
    vector<double> dist(n, 0), val(n), w(n);
    for (int i = 0; i < binN; i++){
      const vector<CostfIdx> &similari = similar[i];
      double totalDist = 0, totoalWeight = 0;
      for (int j = 0; j < n; j++){
        int ithIdx =similari[j].second;
        dist[j] = similari[j].first;
        val[j] = sal[ithIdx];
        w[j] = pW[ithIdx];
        totalDist += dist[j];
        totoalWeight += w[j];
      }
      double valCrnt = 0;
      for (int j = 0; j < n; j++)
        valCrnt += val[j] * (totalDist - dist[j]) * w[j];

      newSal[i] =  valCrnt / (totalDist * totoalWeight);
    }
    normalize(newSal1d, sal1f, 0, 1, NORM_MINMAX, CV_32FC1);
  }

} // end namespace regioncontrast
