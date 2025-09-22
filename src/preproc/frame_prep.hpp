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
#ifndef __PREPROCESS_SINGLE_FRAME_INPUT_HPP_
#define __PREPROCESS_SINGLE_FRAME_INPUT_HPP_

#include "ai_core/algo_data_types.hpp"
#include "ai_core/preproc_base.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {
class FramePreprocess : public IPreprocssPlugin {
public:
  FramePreprocess() = default;
  ~FramePreprocess() = default;

  virtual bool process(const AlgoInput &, const AlgoPreprocParams &,
                       TensorData &,
                       std::shared_ptr<RuntimeContext> &) const override;

  virtual bool batchProcess(const std::vector<AlgoInput> &,
                            const AlgoPreprocParams &, TensorData &,
                            std::shared_ptr<RuntimeContext> &) const override;

private:
  TypedBuffer singleProcess(const FramePreprocessArg &args,
                            const FrameInput &input,
                            FrameTransformContext &runtimeArgs) const;

  TypedBuffer
  batchProcess(const FramePreprocessArg &args,
               const std::vector<FrameInput> &input,
               std::vector<FrameTransformContext> &runtimeArgs) const;
};
} // namespace ai_core::dnn

#endif
