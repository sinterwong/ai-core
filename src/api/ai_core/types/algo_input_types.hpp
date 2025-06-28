/**
 * @file algo_input_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __ALGO_INPUT_TYPES_HPP__
#define __ALGO_INPUT_TYPES_HPP__

#include "infer_common_types.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace ai_core {
struct FramePreprocessArg {
  cv::Rect roi;
  std::vector<float> meanVals;
  std::vector<float> normVals;
  Shape originShape;
  Shape modelInputShape;

  bool needResize = true;
  bool isEqualScale;
  cv::Scalar pad = {0, 0, 0};
  int topPad = 0;
  int leftPad = 0;

  DataType dataType;
  bool hwc2chw = false;
};

struct FrameInput {
  cv::Mat image;
  std::string inputName;
};

} // namespace ai_core

#endif // __ALGO_INPUT_TYPES_HPP__