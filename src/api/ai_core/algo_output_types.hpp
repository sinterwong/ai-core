/**
 * @file algo_output_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __ALGO_OUTPUT_TYPES_HPP__
#define __ALGO_OUTPUT_TYPES_HPP__

#include <array>
#include <memory>
#include <vector>

namespace cv {
template <typename _Tp> class Rect_;
using Rect = Rect_<int>;

class Mat;
} // namespace cv
namespace ai_core {

struct BBox {
  std::shared_ptr<cv::Rect> rect;
  float score;
  int label;
};

struct ClsRet {
  float score;
  int label;
};

struct FeatureRet {
  std::vector<float> feature;
  int featSize;
};

struct FprClsRet {
  float score;
  int label;
  int birad;
  std::vector<float> scoreProbs;
};

struct DetRet {
  std::vector<BBox> bboxes;
};

struct DaulRawSegRet {
  std::shared_ptr<cv::Mat> mask;
  std::shared_ptr<cv::Mat> prob;
  std::shared_ptr<cv::Rect> roi;
  float ratio;
  int leftShift;
  int topShift;
};

struct BDiagSpecRet {
  float malignantScore;
  std::vector<float> feat;
  // breast signs
  float irregularShape;
  float spiculation;
  float blur;
  std::array<float, 6> lesionCls;
  float microlobulation;
  float angularMargins;
  float deeperThanwide;
  float calcification;
};

struct TDiagSpecRet {
  float tirads;
  std::vector<float> feat;
  // thyroid signs
  std::array<float, 5> structure;
  std::array<float, 2> eccentric;
  std::array<float, 3> margin;
  std::array<float, 2> aspectRatio;
  std::array<float, 5> echo;
  std::array<float, 5> focalEcho;
};

} // namespace ai_core

#endif // __ALGO_OUTPUT_TYPES_HPP__