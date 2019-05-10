/*
  Modified by Ming-Ming on Aug. 15th, 2010.

  Copyright (C) 2006 Pedro Felzenszwalb

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SERVER_BASIC_SEGMENT_IMAGE_H_
#define SERVER_BASIC_SEGMENT_IMAGE_H_
#include <opencv2/core/core.hpp>
/* #include <opencv2/highgui/highgui.hpp> */
#include <opencv2/imgproc/imgproc.hpp>
/* #include <opencv2/core/matx.hpp> */
#include <saliency/some_definition.h>


static const int COLOR_NUM = 29, COLOR_NU_NO_GRAY = 24;
const Vec3b gColors[COLOR_NUM] =
{
  Vec3b(0, 0, 255),	  Vec3b(0, 255, 0),		Vec3b(255, 0, 0),     Vec3b(153, 0, 48),	Vec3b(0, 183, 239),
  Vec3b(255, 255, 0),   Vec3b(255, 126, 0),   Vec3b(255, 194, 14),  Vec3b(168, 230, 29),
  Vec3b(237, 28, 36),   Vec3b(77, 109, 243),  Vec3b(47, 54, 153),   Vec3b(111, 49, 152),  Vec3b(156, 90, 60),
  Vec3b(255, 163, 177), Vec3b(229, 170, 122), Vec3b(245, 228, 156), Vec3b(255, 249, 189), Vec3b(211, 249, 188),
  Vec3b(157, 187, 97),  Vec3b(153, 217, 234), Vec3b(112, 154, 209), Vec3b(84, 109, 142),  Vec3b(181, 165, 213),
  Vec3b(40, 40, 40),	  Vec3b(70, 70, 70),	Vec3b(120, 120, 120), Vec3b(180, 180, 180), Vec3b(220, 220, 220)
};

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
 *	pImgInd: index of each components
 */

//"Default: k = 500, sigma = 1.0, min_size = 1000\n") or k = 200, sigma = 0.5, min_size = 50
int SegmentImage(const cv::Mat& _src3f, cv::Mat& pImgInd,
                 double sigma = 0.5, double c = 200, int min_size = 50);
/*
 * return segmented image
 *
 */
int ShowLabel(const cv::Mat& label1i, cv::Mat& imglabel3u,
              int labelNum, bool showIdx = false);

inline Vec3b RandomColor() {
  return Vec3b((uchar)(rand() % 200 + 25), (uchar)(rand() % 200 + 25), (uchar)(rand() % 200 + 25));
}



#endif
