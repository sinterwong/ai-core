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
#ifndef __OPENCV_CPU_GENERIC_CV_PREPROCESSOR_HPP__
#define __OPENCV_CPU_GENERIC_CV_PREPROCESSOR_HPP__

#include "ai_core/algo_input_types.hpp"
#include "ai_core/preproc_types.hpp"
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
  cv::Mat preprocessSingleFrame(const FramePreprocessArg &params,
                                const FrameInput &frameInput,
                                FrameTransformContext &runtimeArgs) const;

  TypedBuffer preprocessFP32(const cv::Mat &normalizedImage, int inputChannels,
                             int inputHeight, int inputWidth,
                             bool hwc2chw) const;

  TypedBuffer preprocessFP16(const cv::Mat &normalizedImage, int inputChannels,
                             int inputHeight, int inputWidth,
                             bool hwc2chw) const;

  void convertLayout(const cv::Mat &image, float *dst, bool hwc2chw) const;
};
} // namespace ai_core::dnn::cpu
#endif