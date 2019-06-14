#include "saliency_region_contrast.h"

#include <iostream>
#include <map>
#include <ctime>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>

#include <basic/block.h>
#include <basic/graph.h>
#include <basic/segment_image.h>
#include <basic/ytplatform.h>
#include <basic/image_operations.h>

using namespace std;
using namespace cv;

namespace regioncontrast {

// 3 result;
// salient region in grayscale
// salient region after binarization
// color image with only salient region
int RegionContrast::ProcessSingleImg(const string& img_path,
                                     string& res_salient,
                                     string& res_salient_bi,
                                     string& res_salient_cut) {


  // cout << "cpp: " << __cplusplus << endl;
  int end_pos = img_path.rfind(".");
  string str_name = img_path.substr(0, end_pos) + "_" + to_string(std::time(0));
  res_salient = str_name + "_RC.png"; // Region contrast
  res_salient_bi = str_name + "_BI.png";
  res_salient_cut = str_name + "_CUT.png";

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
  // save region contrast image
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  imwrite(res_salient, sal1f*255, compression_params);

  //  Binarization
  Mat sal_bi1u, img_cut3f;
  // Parameters for Binarization function
  const double aver_para = 1.85;
  const double max_para = 0.25;
  const bool use_max = false;
  Binarization(sal1f, sal_bi1u, aver_para, max_para, use_max);
  imwrite(res_salient_bi, sal_bi1u*255, compression_params);

  // Mat criterion = imread("/home/acytoo/Pictures/6112.png", 0);
  // criterion /= 255;
  // Mat tmp = criterion - sal_bi1u;
  // Mat tp = criterion - tmp;
  // imshow("tp", tp*255);
  // imshow("sal1f", sal_bi1u*255);
  // waitKey(0);

  // double tp_sum = cv::sum(tp)[0];
  // double recall = tp_sum / cv::sum(criterion)[0];
  // double precision = tp_sum / cv::sum(sal_bi1u)[0];


  // cout << "precision " << precision << " recall " << recall << endl;

  // cout << sal_bi1u << endl;
  CutImage(img3f, sal_bi1u, img_cut3f);
  imwrite(res_salient_cut, img_cut3f*255, compression_params);
  return 0;  //
}



int RegionContrast::ProcessImages(const std::string& root_dir_path, int& amount, int& time_cost,
                                  bool benchmark, double& average_precision, double& average_recall,
                                  double& average_f, double cut_threshold) {
  // cout << "benchmark " << benchmark << endl;
  // double cut_threshold = 1.8; // threshold for colored image cut
  // Read the names of images we are going to process
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
      // cout << "Please rename your file 'cut_result'" << endl;
      return -1;
    }
  }
  else {
    ytfile::mk_dir(saliency_dir_path);
  }
  // Finish make directory; Start cutting

  vector<double> precisions(image_amount, 0.0); // precision of each cut
  vector<double> recalls(image_amount, 0.0);
  vector<double> fprs(image_amount, 0.0);

  // cout << "image_amount " << image_amount << endl;
  // precisions.reserve(image_amount);
  // cout << "precisions capacity " << precisions.capacity() << endl;
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  // max_para is the fixed threshold in Binarization.
  const double aver_para = cut_threshold;
  const double max_para = cut_threshold;
  const bool use_max = true;


  if (benchmark) {
    // Benchmark
#pragma omp parallel for
    for (int i = 0; i < image_amount; ++i){
      string img_path = image_names[i];
      int end_pos = img_path.rfind(".");
      string str_name = img_path.substr(0, end_pos);
      string res_salient = str_name + "_RC.png"; // Region contrast
      string res_salient_bi = str_name + "_BI.png";
      string res_salient_cut = str_name + "_CUT.png";

      Mat img3f = imread(root_dir_path+ "/" + img_path);
      CV_Assert_(img3f.data != NULL, ("Can't load image \n")); // if () continue;
      img3f.convertTo(img3f, CV_32FC3, 1.0/255);

      Mat sal1f = GetRegionContrast(img3f);
      // save region contrast image
      // vector<int> compression_params;  // Sometimes without these parameters we can't save image

      imwrite(saliency_dir_path + "/" + res_salient, sal1f*255, compression_params);

      //  Binarization
      Mat sal_bi1u, img_cut3f;
      // const double aver_para = 1.8;
      // const double max_para = 0.5;
      // const bool use_max = true;
      Binarization(sal1f, sal_bi1u, aver_para, max_para, use_max);
      imwrite(saliency_dir_path + "/" + res_salient_bi, sal_bi1u*255, compression_params);

      // save colored cut
      CutImage(img3f, sal_bi1u, img_cut3f);
      imwrite(saliency_dir_path + "/" + res_salient_cut, img_cut3f*255, compression_params);

      // calculate precession
      Mat criterion1u = imread(root_dir_path+ "/criteria/" + str_name + ".png", 0); // 0 or 255
      criterion1u /= 255;
      double gt_sum = cv::sum(criterion1u)[0];
      Mat tmp = criterion1u - sal_bi1u;
      Mat tp = criterion1u - tmp;
      double tp_sum = cv::sum(tp)[0];
      double p_sum = cv::sum(sal_bi1u)[0];
      double fp_sum = p_sum - tp_sum;
      double fp_tn_sum = sal_bi1u.rows * sal_bi1u.cols - gt_sum;

      recalls[i] = tp_sum / gt_sum; // TPR
      precisions[i] = tp_sum / p_sum;
      fprs[i] = fp_sum / fp_tn_sum;


      // sal_bi1u.setTo(0, criterion1u);
      // cout << cv::sum(sal_bi1u)[0] << endl;
      // precisions[i] = 1 - cv::sum(sal_bi1u)[0] / cv::sum(criterion1u)[0];
      // cout << precisions[i] << endl;
    }
  } else {
    // No benchmark
#pragma omp parallel for
    for (int i = 0; i < image_amount; ++i){
      string img_path = image_names[i];
      int end_pos = img_path.rfind(".");
      string str_name = img_path.substr(0, end_pos) + "_" + to_string(std::time(0));
      string res_salient = str_name + "_RC.png"; // Region contrast
      string res_salient_bi = str_name + "_BI.png";
      string res_salient_cut = str_name + "_CUT.png";

      Mat img3f = imread(root_dir_path+ "/" + img_path);
      CV_Assert_(img3f.data != NULL, ("Can't load image \n")); // if () continue;
      img3f.convertTo(img3f, CV_32FC3, 1.0/255);

      Mat sal1f = GetRegionContrast(img3f);
      // save region contrast image
      // vector<int> compression_params;
      // Sometimes without these parameters we can't save image
      // compression_params.push_back(IMWRITE_PNG_COMPRESSION);
      // compression_params.push_back(9);
      imwrite(saliency_dir_path + "/" + res_salient, sal1f*255, compression_params);

      //  Binarization
      Mat sal_bi1u, img_cut3f;
      Binarization(sal1f, sal_bi1u, aver_para, max_para, use_max);
      imwrite(saliency_dir_path + "/" + res_salient_bi, sal_bi1u*255, compression_params);

      CutImage(img3f, sal_bi1u, img_cut3f);
      imwrite(saliency_dir_path + "/" + res_salient_cut, img_cut3f*255, compression_params);
    }
  }

  amount = image_amount;
  time_cost = (int) std::time(0) - start_time;

  // double average_precision = 0.0;
  double average_fpr = 0.0;
  for (int i=0; i != image_amount; ++i) {
    average_recall += recalls[i];
    average_precision += precisions[i];
    average_fpr += fprs[i];
  }

  average_precision /= image_amount;
  average_recall /= image_amount;
  average_f = average_precision * average_recall * 2 / (average_precision + average_recall);
  average_fpr /= image_amount;
  // cout << average_precision << " " << average_recall << " " << average_f << " latex format " << average_fpr << endl;
  cout << cut_threshold << " & "
       << average_precision*100 << "\\% & "
       << average_recall*100 << "\\% & "
       << average_f*100 << "\\% & "
       << average_fpr << " & "
       << average_recall << " & "
       << time_cost << "s \\\\" << endl
      ;
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
  double sigma_dist = 4; // old value: 0.4
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
  // cout << " region num " << reg_num << endl;
  // cout << reg_sal1dv << endl;
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
  int quantize_num_base[3] = {12, 12, 12};
  float quantize_num[3] = {quantize_num_base[0] - 0.0001f,
                           quantize_num_base[1] - 0.0001f,
                           quantize_num_base[2] - 0.0001f}; // 11.9999, 11.9999, 11.9999
  int color_mask[3] = {quantize_num_base[1] * quantize_num_base[2], quantize_num_base[2], 1}; // 144, 12, 1
  CV_Assert(img3f.data != NULL); // works in opencv 4
  color_idx1i = Mat::zeros(img3f.size(), CV_32S);
  int rows = img3f.rows, cols = img3f.cols;
  if (img3f.isContinuous() && color_idx1i.isContinuous()) { // Accelerate
    cols *= rows;
    rows = 1;
  }

  // Build color pallet{color_identifier:occured_times}
  map<int, int> pallet;
  for (int y = 0; y < rows; ++y) {
    const float* p_ori_img = img3f.ptr<float>(y);
    int* p_color_idx = color_idx1i.ptr<int>(y);
    for (int x = 0; x < cols; ++x, p_ori_img += 3) { // (B*144 + G*12 + r*1) * 11.9999
      p_color_idx[x] = (int)(p_ori_img[0]*quantize_num[0])*color_mask[0] +
                       (int)(p_ori_img[1]*quantize_num[1])*color_mask[1] +
                       (int)(p_ori_img[2]*quantize_num[2])*color_mask[2];
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
  // If the number of color left is less than 10, we don't want to drop any color.
  if (max_num <= 10)
    max_num = min(10, (int)num.size());

  pallet.clear();
  for (int i = 0; i < max_num; ++i)
    pallet[num[i].second] = i;
  // pallet{id:color_precedence}, size = max_num

  // Smooth
  // high frequency color
  vector<Vec3i> color3i(num.size());
  for (unsigned int i = 0, stop = num.size(); i < stop; ++i) {
    color3i[i][0] = num[i].second / color_mask[0];
    color3i[i][1] = num[i].second % color_mask[0] / color_mask[1];
    color3i[i][2] = num[i].second % color_mask[1];
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

  // Histogram
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
      // reg.centroid.x += x;
      // reg.centroid.y += y;
      ++reg_color_freq1i(*p_reg_idx, *p_color_idx);
      reg.ad2c += Point2d(abs(x - cx), abs(y - cy));
    }
  }

  for (int i = 0; i < reg_num; ++i) {
    Region &reg = regs[i];
    // reg.centroid.x /= reg.pix_num * cols; // percentage
    // reg.centroid.y /= reg.pix_num * rows;
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
  const double k_para = -9.0;
  Mat_<double> region_dist_dict1d = Mat::zeros(reg_num, reg_num, CV_64F); // region_num x region_num, double
  reg_sal1dv = Mat::zeros(1, reg_num, CV_64F); // 1 x region_num, double
  double* p_reg_sal = (double*)reg_sal1dv.data;
  for (int i = 0; i < reg_num; ++i) {
    // const Point2d &rc = regs[i].centroid;
    for (int j = 0; j < reg_num; ++j) {
      if(i<j) {
        double dd = 0;
        const vector<CostfIdx> &c1 = regs[i].fre_idx, &c2 = regs[j].fre_idx;
        for (size_t m = 0; m < c1.size(); ++m)
          for (size_t n = 0; n < c2.size(); ++n)
            // Color distance of each region
            dd += color_dist_dict1f[c1[m].second][c2[n].second] * c1[m].first * c2[n].first;

        region_dist_dict1d[j][i] = region_dist_dict1d[i][j] = dd;// * exp(-pntSqrDist(rc, regs[j].centroid)/sigma_dist);
      }
      // cout << "p_reg_sal [i] before += " << p_reg_sal[i] << endl;
      p_reg_sal[i] += regs[j].pix_num * region_dist_dict1d[i][j];
    }
    p_reg_sal[i] *= exp((sqr(regs[i].ad2c.x) + sqr(regs[i].ad2c.y))*k_para);
  }
}

void RegionContrast::Binarization(const Mat &sal1f, Mat &sal_bi1u,
                                  double aver_para, double max_para, bool use_max) {
  CV_Assert(sal1f.type() == CV_32FC1);
  double cut_threshold = 0.0;

  if (use_max) {
    // Fixed threshold
    // double max_value = 0.0;
    // double* p_max_value = &max_value;
    // minMaxLoc(sal1f,NULL, p_max_value);
    // cout << max_value << endl;
    cut_threshold = max_para;
  } else {
    // cut_threshold = aver_para * cv::sum(sal1f)[0]/sal1f.rows/sal1f.cols;
    cut_threshold = aver_para * cv::mean(sal1f)[0];
  }
  cv::threshold(sal1f, sal_bi1u, cut_threshold, 1, THRESH_BINARY);
  sal_bi1u.convertTo(sal_bi1u, CV_8UC1);
}


// img3f: original image
// sal_bi1l: binaried saliency image
// img_cut3f: colored image after cut
void RegionContrast::CutImage(const Mat &img3f, const Mat &sal_bi1u, Mat &img_cut3f) {
  img3f.copyTo(img_cut3f);
  img_cut3f.setTo(0, sal_bi1u);
  img_cut3f = img3f - img_cut3f;
}

} // end namespace regioncontrast
