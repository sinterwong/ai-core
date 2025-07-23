/**
 * @file ncnn_generic_preprocessor.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __NCNN_GENERIC_PREPROCESSOR_HPP_
#define __NCNN_GENERIC_PREPROCESSOR_HPP_

#include "ai_core/algo_input_types.hpp"
#include "ai_core/preproc_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "preproc/frame_preprocessor_base.hpp"

namespace ai_core::dnn::mncnn {
class NcnnGenericPreprocessor : public IFramePreprocessor {
public:
  NcnnGenericPreprocessor() = default;
  ~NcnnGenericPreprocessor() = default;

  TypedBuffer process(FramePreprocessArg &params,
                      const FrameInput &frameInput) const override;
};
} // namespace ai_core::dnn::mncnn
#endif