/**
 * @file generic_frame_preproc_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Shared IPreprocessPlugin adapter for generic frame preprocessing
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_GENERIC_FRAME_PREPROC_BASE_HPP
#define AI_CORE_GENERIC_FRAME_PREPROC_BASE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/tensor_data.hpp"
#include "frame_preprocessor_base.hpp"

namespace ai_core::dnn {

/**
 * @brief Implements the IPreprocessPlugin boilerplate for generic frame
 * preprocessing (param/input validation, TensorData packaging, transform
 * context publication). Concrete plugins provide the pixel kernel.
 */
class GenericFramePreprocBase : public IPreprocessPlugin {
public:
  InferErrorCode process(const AlgoInput &input,
                         const AlgoPreprocParams &params, TensorData &output,
                         std::shared_ptr<RuntimeContext> &runtime_context)
      const final;

  InferErrorCode batchProcess(const std::vector<AlgoInput> &inputs,
                              const AlgoPreprocParams &params,
                              TensorData &output,
                              std::shared_ptr<RuntimeContext> &runtime_context)
      const final;

protected:
  virtual const IFramePreprocessor &kernel() const = 0;
};

} // namespace ai_core::dnn

#endif
