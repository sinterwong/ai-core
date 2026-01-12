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
#ifndef AI_CORE_ALGO_INPUT_TYPES_HPP
#define AI_CORE_ALGO_INPUT_TYPES_HPP

#include <memory>
#include <vector>

namespace cv {
template <typename Tp> class Rect_;
using Rect = Rect_<int>;

class Mat;
} // namespace cv

namespace ai_core {
struct FrameInput {
  std::shared_ptr<cv::Mat> image;
  std::shared_ptr<cv::Rect> input_roi;
};

struct FrameInputWithMask {
  FrameInput frame_input;
  std::vector<cv::Rect> mask_regions;
};

} // namespace ai_core

#endif