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

#include <opencv2/core/types.hpp>
#include <vector> // For std::vector

namespace ai_core {

struct BBox {
  cv::Rect rect;
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

} // namespace ai_core

#endif // __ALGO_OUTPUT_TYPES_HPP__