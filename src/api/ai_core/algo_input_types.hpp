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

#include <memory>

namespace cv {
template <typename _Tp> class Rect_;
using Rect = Rect_<int>;

class Mat;
} // namespace cv

namespace ai_core {
struct FrameInput {
  std::shared_ptr<cv::Mat> image;
  std::shared_ptr<cv::Rect> inputRoi;
};

} // namespace ai_core

#endif // __ALGO_INPUT_TYPES_HPP__