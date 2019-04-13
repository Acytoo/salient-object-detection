#include "segment_graph.h"
#include "segment_image.h"

#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/matx.hpp>


using namespace std;
using namespace cv;

// template<typename T> inline T sqr(T x) { return x * x; } // out of range risk for T = byte, ...

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
 *	im: image to segment.
 *	sigma: to smooth the image.
 *	c: constant for threshold function.
 *	min_size: minimum component size (enforced by post-processing stage).
 *	nu_ccs: number of connected components in the segmentation.
 * Output:
 *	colors: colors assigned to each components
 *	pImgInd: index of each components, [0, colors.size() -1]
 */
int SegmentImage(const Mat& _src3f, Mat& pImgInd,
                 double sigma, double c, int min_size) {
  CV_Assert(_src3f.type() == CV_32FC3);
  int width(_src3f.cols), height(_src3f.rows);
  Mat smImg3f;
  GaussianBlur(_src3f, smImg3f, Size(), sigma, 0, BORDER_REPLICATE);

  // build graph
  edge *edges = new edge[width*height*4];
  int num = 0;
  {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; ++x) {
        if (x < width-1) {
          edges[num].a = y * width + x;
          edges[num].b = y * width + (x+1);
          edges[num].w = diff(smImg3f, x, y, x+1, y);
          num++;
        }

        if (y < height-1) {
          edges[num].a = y * width + x;
          edges[num].b = (y+1) * width + x;
          edges[num].w = diff(smImg3f, x, y, x, y+1);
          num++;
        }

        if ((x < width-1) && (y < height-1)) {
          edges[num].a = y * width + x;
          edges[num].b = (y+1) * width + (x+1);
          edges[num].w = diff(smImg3f, x, y, x+1, y+1);
          num++;
        }

        if ((x < width-1) && (y > 0)) {
          edges[num].a = y * width + x;
          edges[num].b = (y-1) * width + (x+1);
          edges[num].w = diff(smImg3f, x, y, x+1, y-1);
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
  pImgInd.create(smImg3f.size(), CV_32S);

  int idxNum = 0;
  for (int y = 0; y < height; y++) {
    int *imgIdx = pImgInd.ptr<int>(y);
    for (int x = 0; x < width; ++x) {
      int comp = u->find(y * width + x);
      if (marker.find(comp) == marker.end())
        marker[comp] = idxNum++;

      int idx = marker[comp];
      imgIdx[x] = idx;
    }
  }
  delete u;

  return idxNum;
}


int ShowLabel(const cv::Mat& label1i, cv::Mat& img_label3u,
              int labelNum, bool showIdx) {
  bool useRandom = labelNum > 0;
  labelNum = useRandom ? labelNum : COLOR_NU_NO_GRAY;
  vector<Vec3b> colors(labelNum);
  if (useRandom)
    for (size_t i = 0; i < colors.size(); i++)
      colors[i] = RandomColor();
  else
    for (size_t i = 0; i < colors.size(); i++)
      colors[i] = gColors[i];
  img_label3u = Mat::zeros(label1i.size(), CV_8UC3);
  for (int y = 0; y < label1i.rows; y++)	{
    Vec3b* showD = img_label3u.ptr<Vec3b>(y);
    const int* label = label1i.ptr<int>(y);
    for (int x = 0; x < label1i.cols; ++x)
      if (label[x] >= 0){
        showD[x] = colors[label[x] % labelNum];
        if (showIdx)
          showD[x][2] = (byte)(label[x]);
      }
  }
  return 0;
}


