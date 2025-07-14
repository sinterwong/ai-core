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

#include "ai_core/infer_common_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include <memory>

namespace cv {
template <typename _Tp> class Rect_;
template <typename _Tp> class Scalar_;

using Rect = Rect_<int>;
using Scalar = Scalar_<double>;

class Mat;
} // namespace cv

namespace ai_core {

struct FramePreprocessArg {
  enum class FramePreprocType : int8_t {
    OPENCV_CPU_GENERIC = 0, // ROI -> Resize -> Normalize -> Layout convert
    NCNN_GENERIC,
    CUDA_GENERIC,
    OPENCV_CPU_WITH_MASK
  };

  FramePreprocType preprocTaskType = FramePreprocType::OPENCV_CPU_GENERIC;

  std::shared_ptr<cv::Rect> roi;
  std::vector<float> meanVals;
  std::vector<float> normVals;
  Shape originShape;
  Shape modelInputShape;

  bool needResize = true;
  bool isEqualScale;
  std::shared_ptr<cv::Scalar> pad;
  int topPad = 0;
  int leftPad = 0;

  bool hwc2chw = false;

  DataType dataType;
  BufferLocation outputLocation = BufferLocation::CPU;
  std::vector<std::string> inputNames;
};

struct FrameInput {
  std::shared_ptr<cv::Mat> image;
};

} // namespace ai_core

#endif // __ALGO_INPUT_TYPES_HPP__