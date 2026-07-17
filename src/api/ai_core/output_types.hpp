/**
 * @file output_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_ALGO_OUTPUT_TYPES_HPP
#define AI_CORE_ALGO_OUTPUT_TYPES_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/tensor_data.hpp"
#include <map>
#include <memory>
#include <opencv2/core/types.hpp>
#include <vector>

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

using RawModelOutput = TensorData;

struct FprClsRet {
  float score;
  int label;
  int birad;
  std::vector<float> score_probs;
};

struct DetRet {
  std::vector<BBox> bboxes;
};

struct SegRet {
  std::map<int, std::vector<Contour>> cls_to_contours;
};

struct DualRawSegRet {
  std::shared_ptr<cv::Mat> mask;
  std::shared_ptr<cv::Mat> prob;
  std::shared_ptr<cv::Rect> roi;
  float ratio;
  int left_shift;
  int top_shift;
};

struct OCRRecoRet {
  int64_t output_lengths;
  std::vector<int64_t> outputs;
};

} // namespace ai_core

#endif