/**
 * @file cpu_generic_preprocessor.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_OPENCV_CPU_GENERIC_CV_PREPROCESSOR_HPP
#define AI_CORE_OPENCV_CPU_GENERIC_CV_PREPROCESSOR_HPP

#include "ai_core/input_types.hpp"
#include "ai_core/preprocess_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "preproc/frame_preprocessor_base.hpp"
#include "vision_util.hpp"

namespace cv {
class Mat;
} // namespace cv

namespace ai_core::dnn::cpu {
class CpuGenericCvPreprocessor : public IFramePreprocessor {
public:
  explicit CpuGenericCvPreprocessor() {}

  TypedBuffer process(const FramePreprocessArg &, const FrameInput &,
                      FrameTransformContext &) const override;

  TypedBuffer batchProcess(const FramePreprocessArg &,
                           const std::vector<FrameInput> &,
                           std::vector<FrameTransformContext> &) const override;

private:
  // Crop the ROI, convert the pixel format when the conversion cannot be
  // folded into the normalization pass, and resize to the model input size.
  // Stays 8-bit (no per-pixel float work here); returns the prepared image.
  cv::Mat cropAndResize(const FramePreprocessArg &params,
                        const FrameInput &frame_input,
                        const utils::PixelFormatPlan &format_plan,
                        FrameTransformContext &runtime_args) const;

  // Single-pass fusion of channel reordering, normalization
  // ((v - mean)/std), dtype cast (fp32/fp16) and layout (HWC/CHW), writing
  // directly into a fresh TypedBuffer. `dst_offset_elems` places the frame
  // inside a batch buffer.
  void writeNormalizedLayout(const cv::Mat &prepared_u8,
                             const FramePreprocessArg &params,
                             const utils::PixelFormatPlan &format_plan,
                             TypedBuffer &dst, size_t dst_offset_elems) const;
};
} // namespace ai_core::dnn::cpu
#endif
