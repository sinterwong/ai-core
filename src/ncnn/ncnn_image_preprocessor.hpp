/**
 * @file ncnn_image_preprocessor.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __NCNN_INFERENCE_FRAME__DET_HPP_
#define __NCNN_INFERENCE_FRAME__DET_HPP_

#include "ai_core/types/algo_data_types.hpp"
#include "ai_core/types/typed_buffer.hpp"
#include <memory>

namespace ai_core::dnn::ncnn {
class ImagePreprocessor {
public:
  explicit ImagePreprocessor() {}

  TypedBuffer process(FramePreprocessArg &params,
                      const FrameInput &frameInput) const;
};
} // namespace ai_core::dnn::ncnn
#endif