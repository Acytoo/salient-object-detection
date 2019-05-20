#include "saliency_region_contrast.h"

#include <iostream>
#include <map>
#include <ctime>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>

// #include <saliency/saliency_cut.h>
#include <basic/block.h>
#include <basic/graph.h>
#include <basic/segment_image.h>
#include <basic/ytplatform.h>
#include <basic/image_operations.h>

using namespace std;
using namespace cv;

namespace regioncontrast {

  int RegionContrast::ProcessSingleImg(const string& img_path,
                                    string& result_rc_path) {
    cout << "cpp: " << __cplusplus << endl;
    int end_pos = img_path.rfind(".");
    string str_name = img_path.substr(0, end_pos) + "_" + to_string(std::time(0));
    result_rc_patbh = str_name + "_RC.png"; // Region contrast
    // result_rcc_path = str_name + "_RCC.png"; // Region contrast cut
    //first error detection, permission and presense of the image
    //imread 2nd parameter 0: grayscale; 1: 3 channels; -1: as is(with alpha channel)
    Mat img3f = imread(img_path, 1); // 3u now
    if (!img3f.data) {
      cout << "empty image" << endl;
      return -1;
    }
    // CV_Assert(img3f.type() == CV_8UC3);
    //convert to float, 3 channels
    img3f.convertTo(img3f, CV_32FC3, 1.0/255, 0);


    Mat sal1f = GetRegionContrast(img3f);

    //  Binarization
    cv::threshold(sal1f, sal1f, cv::sum(sal1f)[0]/sal1f.rows/sal1f.cols/0.5, 1, THRESH_BINARY);

    // save region contrast image
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    imwrite(result_rc_path, sal1f*255, compression_params);
    Mat sal_bi1f;
    // Binarization(sal1f, sal_bi1f);

    return 0;
  }


  int RegionContrast::ProcessImages(const std::string& root_dir_path, int& amount, int& time_cost) {

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
      // string result_rcc_path = str_name + "_RCC.png"; // Region contrast cut
      // printf("OpenMP Test, thread index: %d\n", omp_get_thread_num());

      Mat img3f = imread(root_dir_path+ "/" + img_path);
      CV_Assert_(img3f.data != NULL, ("Can't load image \n"));
      img3f.convertTo(img3f, CV_32FC3, 1.0/255);
      Mat sal = regioncontrast::RegionContrast::GetRegionContrast(img3f);
      imwrite(saliency_dir_path + "/" + result_rc_path, sal*255);
    }
    amount = image_amount;
    time_cost = (int) std::time(0) - start_time;
    return image_amount;
  }


  void RegionContrast::ShowImageInfo(const Mat& img) {
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

    cv::Vec3d vec3d = img.at<cv::Vec3d>(0,0);
    double vec3d0 = img.at<cv::Vec3d>(0,0)[0];
    double vec3d1 = img.at<cv::Vec3d>(0,0)[1];
    double vec3d2 = img.at<cv::Vec3d>(0,0)[2];
    std::cout<<"vec3d = "<<vec3d<<std::endl;

  }


  Mat RegionContrast::GetRegionContrast(const cv::Mat& img3f){
    // basic parameters
    // Larger values of sigma_dist reduce the effect of spatial weighting
    // so contrast to farther regions would contribute more to the
    // saliency of the current region
    double sigma_dist = 0.3; // old value: 0.4
    double seg_k = 50;
    int seg_min_size = 200;
    double seg_sigma = 0.5;
    Mat img_lab3f, region_idx1i;

    // segment image, Lab
    cvtColor(img3f, img_lab3f, COLOR_BGR2Lab);
    int reg_num = SegmentImage(img_lab3f, region_idx1i,
                               seg_sigma, seg_k, seg_min_size);

    // Color quantization, BGR
    Mat color_idx1i, reg_sal1dv, tmp, color3fv;
    int quantize_num = Quantize(img3f, color_idx1i, color3fv, tmp);
    if (quantize_num == 2){
      Mat sal;
      compare(color_idx1i, 1, sal, CMP_EQ);
      sal.convertTo(sal, CV_32F, 1.0/255);
      return sal;
    }
    if (quantize_num <= 1)
      return Mat::zeros(img3f.size(), CV_32F);

    // Build region, Lab
    cvtColor(color3fv, color3fv, COLOR_BGR2Lab);
    vector<Region> regs(reg_num);
    BuildRegions(region_idx1i, regs, color_idx1i, color3fv.cols);
    RegionContrastCore(regs, color3fv, reg_sal1dv, sigma_dist);
    //reg_sal1dv : 1 x region_num, 1 channel double, indicate the saliency value of each region

    Mat sal1f = Mat::zeros(img3f.size(), CV_32F); // greyscale salient image
    cv::normalize(reg_sal1dv, reg_sal1dv, 0, 1, NORM_MINMAX, CV_32F); // normalize the saliency value
    float* p_reg_sal = (float*)reg_sal1dv.data; // convert double to float
    for (int r = 0; r < img3f.rows; ++r) {
      const int* p_reg_idx = region_idx1i.ptr<int>(r);
      float* p_sal = sal1f.ptr<float>(r);
      for (int c = 0; c < img3f.cols; ++c)
        p_sal[c] = p_reg_sal[p_reg_idx[c]];
    }
    GaussianBlur(sal1f, sal1f, Size(3, 3), 0);
    return sal1f;
  }

  // img3f: bgr image, 3 channel float, row x col
  // color_idx1i: color index, same color has same index, 1 channel int, row x col
  // res_color3f: colors after quantize, bgr color, 3 channel float, 1 x col
  // res_color_num: number of each color, 1 channel int, 1 x col
  // ratio: quantize ratio
  int RegionContrast::Quantize(const cv::Mat &img3f,
                               cv::Mat &color_idx1i,
                               cv::Mat &res_color3f,
                               cv::Mat &res_color_num,
                               double ratio) {
    int color_mask_base[3] = {12, 12, 12};
    float color_masks[3] = {color_mask_base[0] - 0.0001f,
                            color_mask_base[1] - 0.0001f,
                            color_mask_base[2] - 0.0001f}; // 11.9999, 11.9999, 11.9999
    int w[3] = {color_mask_base[1] * color_mask_base[2], color_mask_base[2], 1}; // 144, 12, 1
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
        p_color_idx[x] = (int)(p_ori_img[0]*color_masks[0])*w[0] +
          (int)(p_ori_img[1]*color_masks[1])*w[1] +
          (int)(p_ori_img[2]*color_masks[2]);
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
    int rows = region_idx1i.rows, cols = region_idx1i.cols, reg_num = regs.size();
    double cx = cols / 2.0, cy = rows / 2.0;
    Mat_<int> reg_color_freq1i  = Mat_<int>::zeros(reg_num, color_num); // region color frequency
    for (int y = 0; y < rows; ++y) {
      const int *p_reg_idx = region_idx1i.ptr<int>(y);
      const int *p_color_idx = color_idx1i.ptr<int>(y);
      for (int x = 0; x < cols; ++x, ++p_reg_idx, ++p_color_idx) {
        Region &reg = regs[*p_reg_idx];
        ++reg.pix_num;
        reg.centroid.x += x;
        reg.centroid.y += y;
        ++reg_color_freq1i(*p_reg_idx, *p_color_idx);
        reg.ad2c += Point2d(abs(x - cx), abs(y - cy));
      }
    }

    for (int i = 0; i < reg_num; ++i) {
      Region &reg = regs[i];
      reg.centroid.x /= reg.pix_num * cols; // percentage
      reg.centroid.y /= reg.pix_num * rows;
      reg.ad2c.x /= reg.pix_num * cols;
      reg.ad2c.y /= reg.pix_num * rows;
      int *p_reg_color_freq = reg_color_freq1i.ptr<int>(i);
      for (int j = 0; j < color_num; ++j) {
        float fre = (float)p_reg_color_freq[j]/(float)reg.pix_num;
        // if (p_reg_color_freq[j] > EPS)
        if (fre > EPS) // fre > 0
          reg.fre_idx.push_back(make_pair(fre, j));
      }
    }
  }

  void RegionContrast::RegionContrastCore(const vector<Region> &regs,
                                          const cv::Mat &color3fv,
                                          cv::Mat &reg_sal1dv,
                                          double sigma_dist) {
    // Color distance of each color in the image.
    Mat_<float> color_dist_dict1f = Mat::zeros(color3fv.cols, color3fv.cols, CV_32F); // color_size x color_size, float
    Vec3f* p_color = (Vec3f*)color3fv.data;
    for(int i = 0; i < color_dist_dict1f.rows; ++i)
      for(int j = i+1; j < color_dist_dict1f.cols; ++j)
        color_dist_dict1f[i][j] = color_dist_dict1f[j][i] = vecDist<float, 3>(p_color[i], p_color[j]); // Lab color
    // Region distance
    int reg_num = (int)regs.size();
    Mat_<double> region_dist_dict1d = Mat::zeros(reg_num, reg_num, CV_64F); // region_num x region_num, double
    reg_sal1dv = Mat::zeros(1, reg_num, CV_64F); // 1 x region_num, double
    double* p_reg_sal = (double*)reg_sal1dv.data;
    for (int i = 0; i < reg_num; ++i) {
      const Point2d &rc = regs[i].centroid;
      for (int j = 0; j < reg_num; ++j) {
        if(i<j) {
          double dd = 0;
          const vector<CostfIdx> &c1 = regs[i].fre_idx, &c2 = regs[j].fre_idx;
          for (size_t m = 0; m < c1.size(); ++m)
            for (size_t n = 0; n < c2.size(); ++n)
              dd += color_dist_dict1f[c1[m].second][c2[n].second] * c1[m].first * c2[n].first;
          region_dist_dict1d[j][i] = region_dist_dict1d[i][j] = dd * exp(-pntSqrDist(rc, regs[j].centroid)/sigma_dist);
        }
        p_reg_sal[i] += regs[j].pix_num * region_dist_dict1d[i][j];
      }
      p_reg_sal[i] *= exp(-9.0 * (sqr(regs[i].ad2c.x) + sqr(regs[i].ad2c.y)));
    }
  }

  // sal1f: saliency image in 1 channel float
  // sal_bi1f: saliency image after binarization, 1 channel float
  // void RegionContrast::Binarization(const Mat &sal1f, Mat &sal_bi1f) {
  //   CV_Assert(sal1f.type() == CV_32FC1); // 1 channel, so sum()[0] get a double
  //   int row = sal1f.rows, col = sal1f.cols;
  //   sal_bi1f = Mat::zeros(row, col, CV_32FC1);
  //   double threshold = cv::sum(sal1f)[0] / row / col;
  //   // cout << threshold << endl;

  // }

} // end namespace regioncontrast
