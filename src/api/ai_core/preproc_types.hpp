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

#ifndef AI_CORE_PREPROCESS_TYPES_HPP
#define AI_CORE_PREPROCESS_TYPES_HPP

#include "ai_core/typed_buffer.hpp"
#include <string>
#include <vector>

namespace cv {
template <typename Tp> class Rect_;
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
    OpencvCpuGeneric = 0,
    NcnnGeneric,
    CudaGpuGeneric,

    // ROI -> Concat -> Resize -> Normalize -> Layout convert
    OpencvCpuConcatMask
  };
  FramePreprocType preproc_task_type = FramePreprocType::OpencvCpuGeneric;
  Shape model_input_shape;
  bool need_resize;
  bool is_equal_scale;
  std::vector<int> pad;
  std::vector<float> mean_vals;
  std::vector<float> norm_vals;
  bool hwc2chw;
  DataType data_type;
  BufferLocation output_location = BufferLocation::CPU;
  std::vector<std::string> input_names;
};

struct FrameTransformContext {
  bool is_equal_scale;
  Shape origin_shape;
  Shape model_input_shape;
  std::shared_ptr<cv::Rect> roi;
  int top_pad = 0;
  int left_pad = 0;
};

} // namespace ai_core

#endif