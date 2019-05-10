#ifndef SERVER_SALIENCY_SOME_DEFINITION_H_
#define SERVER_SALIENCY_SOME_DEFINITION_H_
#include <complex>
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace std;
// #ifdef max
// #undef max
// #endif

// #ifdef min
// #undef min
// #endif

typedef unsigned char byte; // not working in c++ 17

extern Point const DIRECTION8[9];


extern double const SQRT2;

typedef pair<float, int> CostfIdx;

const double EPS = 1e-200;		// Epsilon (zero value)

#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)

#define ForPoints2(pnt, xS, yS, xE, yE)	for (Point pnt(0, (yS)); pnt.y != (yE); pnt.y++) for (pnt.x = (xS); pnt.x != (xE); pnt.x++)

template<typename T> inline T sqr(T x) { return x * x; } // out of range risk for T = byte, ...
template<class T> inline T pntSqrDist(const Point_<T> &p1, const Point_<T> &p2) {return sqr(p1.x - p2.x) + sqr(p1.y - p2.y);} // out of range risk for T = byte, ...
template<class T> inline double pntDist(const Point_<T> &p1, const Point_<T> &p2) {return sqrt((double)pntSqrDist(p1, p2));} // out of range risk for T = byte, ...
template<class T, int D> inline T vecSqrDist(const Vec<T, D> &v1, const Vec<T, D> &v2) {T s = 0; for (int i=0; i<D; i++) s += sqr(v1[i] - v2[i]); return s;} // out of range risk for T = byte, ...
template<class T, int D> inline T vecDist(const Vec<T, D> &v1, const Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); } // out of range risk for T = byte, ...

#define CV_Assert_(expr, args)                                          \
  {                                                                     \
	if(!(expr)) {                                                       \
      string msg = cv::format args;                                     \
      printf("%s in %s:%d\n", msg.c_str(), __FILE__, __LINE__);         \
      cv::error(cv::Exception(cv::Error::Code::StsAssert, msg, __FUNCTION__, __FILE__, __LINE__) ); } \
  }

#endif
