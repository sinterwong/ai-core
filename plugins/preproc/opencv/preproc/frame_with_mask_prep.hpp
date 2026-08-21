#ifndef AI_CORE_PREPROCESS_SINGLE_FRAME_WITH_MASK_INPUT_HPP
#define AI_CORE_PREPROCESS_SINGLE_FRAME_WITH_MASK_INPUT_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {
class FrameWithMaskPreprocess : public IPreprocessPlugin {
public:
  FrameWithMaskPreprocess() = default;
  ~FrameWithMaskPreprocess() = default;

  virtual InferErrorCode
  process(const AlgoInput &, const AlgoPreprocParams &, TensorData &,
          std::shared_ptr<RuntimeContext> &) const override;

  virtual InferErrorCode
  batchProcess(const std::vector<AlgoInput> &, const AlgoPreprocParams &,
               TensorData &, std::shared_ptr<RuntimeContext> &) const override;
};
} // namespace ai_core::dnn

#endif
