/**
 * @file frame_prep.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_PREPROCESS_SINGLE_FRAME_INPUT_HPP
#define AI_CORE_PREPROCESS_SINGLE_FRAME_INPUT_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {
class FramePreprocess : public IPreprocessPlugin {
public:
  FramePreprocess() = default;
  ~FramePreprocess() = default;

  virtual InferErrorCode process(const AlgoInput &, const AlgoPreprocParams &,
                       TensorData &,
                       std::shared_ptr<RuntimeContext> &) const override;

  virtual InferErrorCode batchProcess(const std::vector<AlgoInput> &,
                            const AlgoPreprocParams &, TensorData &,
                            std::shared_ptr<RuntimeContext> &) const override;

private:
  TypedBuffer singleProcess(const FramePreprocessArg &args,
                            const FrameInput &input,
                            FrameTransformContext &runtime_args) const;

  TypedBuffer
  batchProcess(const FramePreprocessArg &args,
               const std::vector<FrameInput> &input,
               std::vector<FrameTransformContext> &runtime_args) const;
};
} // namespace ai_core::dnn

#endif
