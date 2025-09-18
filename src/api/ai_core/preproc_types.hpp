/**
 * @file preproc_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __PREPROCESS_TYPES_HPP__
#define __PREPROCESS_TYPES_HPP__

#include "ai_core/typed_buffer.hpp"
#include <string>
#include <vector>

namespace cv {
template <typename _Tp> class Rect_;
using Rect = Rect_<int>;
} // namespace cv

namespace ai_core {

struct Shape {
  int w;
  int h;
  int c;
};

struct FramePreprocessArg {
  enum class FramePreprocType : int8_t {
    // ROI -> Resize -> Normalize -> Layout convert
    OPENCV_CPU_GENERIC = 0,
    NCNN_GENERIC,
    CUDA_GPU_GENERIC,
    BATCH_OPENCV_CPU_GENERIC,
    BATCH_CUDA_GPU_GENERIC,

    // ROI -> Concat -> Resize -> Normalize -> Layout convert
    OPENCV_CPU_CONCAT_MASK,
  };
  FramePreprocType preprocTaskType = FramePreprocType::OPENCV_CPU_GENERIC;
  Shape modelInputShape;
  bool needResize;
  bool isEqualScale;
  std::vector<int> pad;
  std::vector<float> meanVals;
  std::vector<float> normVals;
  bool hwc2chw;
  DataType dataType;
  BufferLocation outputLocation = BufferLocation::CPU;
  std::vector<std::string> inputNames;
};

struct FrameTransformContext {
  bool isEqualScale;
  Shape originShape;
  Shape modelInputShape;
  std::shared_ptr<cv::Rect> roi;
  int topPad;
  int leftPad;
};

} // namespace ai_core

#endif // __PREPROCESS_TYPES_HPP__