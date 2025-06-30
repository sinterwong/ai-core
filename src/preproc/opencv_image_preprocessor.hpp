/**
 * @file cpu_image_preprocessor.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __OPENCV_IMAGE_PREPROCESSOR_HPP_
#define __OPENCV_IMAGE_PREPROCESSOR_HPP_

#include "ai_core/algo_input_types.hpp"
#include "ai_core/typed_buffer.hpp"

namespace cv {
class Mat;
} // namespace cv

// TODO: will add the "ort" namespace
namespace ai_core::dnn::cpu {
class ImagePreprocessor {
public:
  explicit ImagePreprocessor() {}

  TypedBuffer process(FramePreprocessArg &params,
                      const FrameInput &frameInput) const;

private:
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