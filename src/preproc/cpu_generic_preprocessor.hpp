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
#include "frame_preprocessor_base.hpp"

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
  // Crop the ROI and resize to the model input size, staying 8-bit (no
  // per-pixel float work here). Returns the prepared uint8 image.
  cv::Mat cropAndResize(const FramePreprocessArg &params,
                        const FrameInput &frame_input,
                        FrameTransformContext &runtime_args) const;

  // Single-pass fusion of normalization ((v - mean)/norm), dtype cast
  // (fp32/fp16) and layout (HWC/CHW), writing directly into a fresh
  // TypedBuffer. `dst_offset_elems` places the frame inside a batch buffer.
  void writeNormalizedLayout(const cv::Mat &prepared_u8,
                             const FramePreprocessArg &params, TypedBuffer &dst,
                             size_t dst_offset_elems) const;
};
} // namespace ai_core::dnn::cpu
#endif