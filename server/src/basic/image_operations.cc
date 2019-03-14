#include "image_operations.h"

#include <iostream>
#include <vector>
#include <queue>
#include <list>

using namespace std;
using namespace cv;
Mat imageoperations::ImageOperations::GetLargestSumNoneZeroRegion(const Mat& mask1u,
                                                                  double ignore_ratio) {
  CV_Assert(mask1u.type() == CV_8UC1 && mask1u.data != NULL);
  ignore_ratio *= mask1u.rows * mask1u.cols * 255;
  Mat_<int> regIdx1i;
  vector<int> index_sum;
  Mat res_mask;
  imageoperations::ImageOperations::GetNoneZeroRegions(mask1u, regIdx1i, index_sum);
  if (index_sum.size() >= 1 && index_sum[0] > ignore_ratio)
    compare(regIdx1i, 0, res_mask, CMP_EQ);
  return res_mask;
}

Rect imageoperations::ImageOperations::GetMaskRange(const Mat& mask1u, int ext, int thresh) {
  int maxX = INT_MIN, maxY = INT_MIN, minX = INT_MAX, minY = INT_MAX, rows = mask1u.rows, cols = mask1u.cols;
  for (int r = 0; r < rows; r++)	{
    const byte* data = mask1u.ptr<byte>(r);
    for (int c = 0; c < cols; c++)
      if (data[c] > thresh) {
        maxX = max(maxX, c);
        minX = min(minX, c);
        maxY = max(maxY, r);
        minY = min(minY, r);
      }
  }

  maxX = maxX + ext + 1 < cols ? maxX + ext + 1 : cols;
  maxY = maxY + ext + 1 < rows ? maxY + ext + 1 : rows;
  minX = minX - ext > 0 ? minX - ext : 0;
  minY = minY - ext > 0 ? minY - ext : 0;

  return Rect(minX, minY, maxX - minX, maxY - minY);
}


// Get continuous components for non-zero labels. Return region index mat (region index
// of each mat position) and sum of label values in each region
int imageoperations::ImageOperations::GetNoneZeroRegions(const Mat_<byte> &label1u,
                                                         Mat_<int> &regIdx1i, vecI &idxSum) {
  vector<pair<int, int>> counterIdx;
  int _w = label1u.cols, _h = label1u.rows, maxIdx = -1;
  regIdx1i.create(label1u.size());
  regIdx1i = -1;

  for (int y = 0; y < _h; y++){
    int *regIdx = regIdx1i.ptr<int>(y);
    const byte *label = label1u.ptr<byte>(y);
    for (int x = 0; x < _w; x++) {
      if (regIdx[x] != -1 || label[x] == 0)
        continue;

      pair<int, int> counterReg(0, ++maxIdx); // Number of pixels in region with index maxIdx
      Point pt(x, y);
      queue<Point, list<Point>> neighbs;
      regIdx[x] = maxIdx;
      neighbs.push(pt);

      // Repeatably add pixels to the queue to construct neighbor regions
      while(neighbs.size()){
        // Mark current pixel
        pt = neighbs.front();
        neighbs.pop();
        counterReg.first += label1u(pt);

        // Mark its unmarked neighbor pixels if similar
        Point nPt(pt.x, pt.y - 1); //Upper
        if (nPt.y >= 0 && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
          regIdx1i(nPt) = maxIdx;
          neighbs.push(nPt);
        }

        nPt.y = pt.y + 1; // lower
        if (nPt.y < _h && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
          regIdx1i(nPt) = maxIdx;
          neighbs.push(nPt);
        }

        nPt.y = pt.y, nPt.x = pt.x - 1; // Left
        if (nPt.x >= 0 && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
          regIdx1i(nPt) = maxIdx;
          neighbs.push(nPt);
        }

        nPt.x = pt.x + 1;  // Right
        if (nPt.x < _w && regIdx1i(nPt) == -1 && label1u(nPt) > 0)	{
          regIdx1i(nPt) = maxIdx;
          neighbs.push(nPt);
        }
      }

      // Add current region to regions
      counterIdx.push_back(counterReg);
    }
  }
  sort(counterIdx.begin(), counterIdx.end(), greater<pair<int, int>>());
  int idxNum = (int)counterIdx.size();
  vector<int> newIdx(idxNum);
  idxSum.resize(idxNum);
  for (int i = 0; i < idxNum; i++){
    idxSum[i] = counterIdx[i].first;
    newIdx[counterIdx[i].second] = i;
  }

  for (int y = 0; y < _h; y++){
    int *regIdx = regIdx1i.ptr<int>(y);
    for (int x = 0; x < _w; x++)
      if (regIdx[x] >= 0)
        regIdx[x] = newIdx[regIdx[x]];
  }
  return idxNum;
}
