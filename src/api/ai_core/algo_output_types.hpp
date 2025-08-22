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

#include "ai_core/infer_common_types.hpp"
#include <map>
#include <memory>
#include <vector>

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

struct SegRet {
  std::map<int, std::vector<Contour>> clsToContours;
};

struct DaulRawSegRet {
  std::shared_ptr<cv::Mat> mask;
  std::shared_ptr<cv::Mat> prob;
  std::shared_ptr<cv::Rect> roi;
  float ratio;
  int leftShift;
  int topShift;
};

} // namespace ai_core

#endif // __ALGO_OUTPUT_TYPES_HPP__