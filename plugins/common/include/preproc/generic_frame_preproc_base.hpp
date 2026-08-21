#ifndef AI_CORE_GENERIC_FRAME_PREPROC_BASE_HPP
#define AI_CORE_GENERIC_FRAME_PREPROC_BASE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/tensor_data.hpp"
#include "preproc/frame_preprocessor_base.hpp"

namespace ai_core::dnn {

/**
 * @brief Implements the IPreprocessPlugin boilerplate for generic frame
 * preprocessing (param/input validation, TensorData packaging, transform
 * context publication). Concrete plugins provide the pixel kernel.
 */
class GenericFramePreprocBase : public IPreprocessPlugin {
public:
  InferErrorCode
  process(const AlgoInput &input, const AlgoPreprocParams &params,
          TensorData &output,
          std::shared_ptr<RuntimeContext> &runtime_context) const final;

  InferErrorCode
  batchProcess(const std::vector<AlgoInput> &inputs,
               const AlgoPreprocParams &params, TensorData &output,
               std::shared_ptr<RuntimeContext> &runtime_context) const final;

protected:
  virtual const IFramePreprocessor &kernel() const = 0;
};

} // namespace ai_core::dnn

#endif
