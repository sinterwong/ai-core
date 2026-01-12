/**
 * @file frame_preprocessor_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_I_FRAME_PREPROCESSOR_HPP
#define AI_CORE_I_FRAME_PREPROCESSOR_HPP

#include "ai_core/algo_input_types.hpp"
#include "ai_core/preproc_types.hpp"
#include "ai_core/typed_buffer.hpp"

namespace ai_core {
/**
 * @brief Interface for frame preprocessors.
 *
 */
class IFramePreprocessor {
public:
  virtual ~IFramePreprocessor() = default;

  virtual TypedBuffer process(const FramePreprocessArg &args,
                              const FrameInput &input,
                              FrameTransformContext &runtime_args) const = 0;

  virtual TypedBuffer
  batchProcess(const FramePreprocessArg &args,
               const std::vector<FrameInput> &input,
               std::vector<FrameTransformContext> &runtime_args) const = 0;
};

} // namespace ai_core

#endif
