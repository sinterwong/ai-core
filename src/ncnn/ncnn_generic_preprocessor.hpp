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
#ifndef AI_CORE_NCNN_GENERIC_PREPROCESSOR_HPP
#define AI_CORE_NCNN_GENERIC_PREPROCESSOR_HPP

#include "ai_core/input_types.hpp"
#include "ai_core/preprocess_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "preproc/frame_preprocessor_base.hpp"

namespace ai_core::dnn::mncnn {
class NcnnGenericPreprocessor : public IFramePreprocessor {
public:
  NcnnGenericPreprocessor() = default;
  ~NcnnGenericPreprocessor() = default;

  TypedBuffer process(const FramePreprocessArg &, const FrameInput &,
                      FrameTransformContext &) const override;

  TypedBuffer batchProcess(const FramePreprocessArg &,
                           const std::vector<FrameInput> &,
                           std::vector<FrameTransformContext> &) const override;
};
} // namespace ai_core::dnn::mncnn
#endif