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

#include <opencv2/opencv.hpp>

namespace infer {
struct FramePreprocessArg {
  cv::Rect roi;
  std::vector<float> meanVals;
  std::vector<float> normVals;
  Shape originShape;

  bool needResize = true;
  bool isEqualScale;
  cv::Scalar pad = {0, 0, 0};
  int topPad = 0;
  int leftPad = 0;
};

struct FrameInput {
  cv::Mat image;
  FramePreprocessArg args;
};

} // namespace infer

#endif // __ALGO_INPUT_TYPES_HPP__