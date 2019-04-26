#include "segment_graph.h"
#include "segment_image.h"

#include <iostream>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/matx.hpp>

using namespace std;
using namespace cv;

// dissimilarity measure between pixels
static inline float diff(const Mat& img3f, int x1, int y1, int x2, int y2) {
  const Vec3f& p1 = img3f.at<Vec3f>(y1, x1);
  const Vec3f& p2 = img3f.at<Vec3f>(y2, x2);
  return sqrt(sqr(p1[0] - p2[0]) + sqr(p1[1] - p2[1]) + sqr(p1[2] - p2[2]));
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * Input:
 *	ori_img3f: image to segment.
 *	sigma: to smooth the image.
 *	c: constant for threshold function.
 *	min_size: minimum component size (enforced by post-processing stage).
 *	nu_ccs: number of connected components in the segmentation.
 * Output:
 *	colors: colors assigned to each components
 *	img_part_idx: index of each components, [0, colors.size() -1]
 */
int SegmentImage(const Mat& ori_img3f, Mat& img_part_idx,
                 double sigma, double c, int min_size) {
  CV_Assert(ori_img3f.type() == CV_32FC3);
  int width(ori_img3f.cols), height(ori_img3f.rows);
  Mat smoothed_img3f;
  GaussianBlur(ori_img3f, smoothed_img3f, Size(), sigma, 0, BORDER_REPLICATE);

  // build graph
  edge *edges = new edge[width*height*4];
  int num = 0;
  {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (x < width-1) {
          edges[num].a = y * width + x;
          edges[num].b = y * width + (x+1);
          edges[num].w = diff(smoothed_img3f, x, y, x+1, y);
          num++;
        }

        if (y < height-1) {
          edges[num].a = y * width + x;
          edges[num].b = (y+1) * width + x;
          edges[num].w = diff(smoothed_img3f, x, y, x, y+1);
          num++;
        }

        if ((x < width-1) && (y < height-1)) {
          edges[num].a = y * width + x;
          edges[num].b = (y+1) * width + (x+1);
          edges[num].w = diff(smoothed_img3f, x, y, x+1, y+1);
          num++;
        }

        if ((x < width-1) && (y > 0)) {
          edges[num].a = y * width + x;
          edges[num].b = (y-1) * width + (x+1);
          edges[num].w = diff(smoothed_img3f, x, y, x+1, y-1);
          num++;
        }
      }
    }
  }

  // segment
  universe *u = segment_graph(width*height, num, edges, (float)c);

  // post process small components
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }
  delete [] edges;

  // pick random colors for each component
  map<int, int> marker;
  img_part_idx.create(smoothed_img3f.size(), CV_32S);

  int idx_num = 0;
  for (int y = 0; y < height; ++y) {
    int *p_img_idx = img_part_idx.ptr<int>(y);
    for (int x = 0; x < width; ++x) {
      int comp = u->find(y * width + x);
      if (marker.find(comp) == marker.end())
        marker[comp] = idx_num++;

      int idx = marker[comp];
      p_img_idx[x] = idx;
    }
  }
  delete u;

  return idx_num;
}


int ShowLabel(const cv::Mat& label1i, cv::Mat& img_label3u,
              int label_num, bool showIdx) {
  bool use_random = label_num > 0;
  label_num = use_random ? label_num : COLOR_NU_NO_GRAY;
  vector<Vec3b> colors(label_num);
  if (use_random)
    for (size_t i = 0; i < colors.size(); i++)
      colors[i] = RandomColor();
  else
    for (size_t i = 0; i < colors.size(); i++)
      colors[i] = gColors[i];
  img_label3u = Mat::zeros(label1i.size(), CV_8UC3);
  for (int y = 0; y < label1i.rows; ++y)	{
    Vec3b* showD = img_label3u.ptr<Vec3b>(y);
    const int* label = label1i.ptr<int>(y);
    for (int x = 0; x < label1i.cols; ++x)
      if (label[x] >= 0){
        showD[x] = colors[label[x] % label_num];
        if (showIdx)
          showD[x][2] = (byte)(label[x]);
      }
  }
  return 0;
}


