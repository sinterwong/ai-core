#ifndef AI_CORE_ALGO_PREPROC_HPP
#define AI_CORE_ALGO_PREPROC_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

/**
 * @brief Standalone preprocessing stage (bypasses the AlgoInference facade).
 *
 * @par Thread safety
 * `process()` and `batchProcess()` may run concurrently after initialization;
 * call-local scratch travels through `TensorData` and `RuntimeContext`.
 * `initialize()` and `terminate()` require exclusive access.
 */
class AlgoPreproc {
public:
  AlgoPreproc(const std::string &module_name);

  ~AlgoPreproc();

  /**
   * @brief Create the plugin and bind validated default parameters.
   */
  InferErrorCode initialize(const AlgoPreprocParams &preproc_params);

  /**
   * @brief Convert one algorithm input into model input tensors.
   *
   * @param preproc_override Per-call replacement for the bound parameters, or
   * `nullptr` to use the defaults from `initialize()`.
   */
  InferErrorCode process(const AlgoInput &input, TensorData &model_input,
                         std::shared_ptr<RuntimeContext> &runtime_context,
                         const AlgoPreprocParams *preproc_override = nullptr);

  /** Convert a batch into the tensor layout required by the plugin. */
  InferErrorCode
  batchProcess(const std::vector<AlgoInput> &input, TensorData &model_input,
               std::shared_ptr<RuntimeContext> &runtime_context,
               const AlgoPreprocParams *preproc_override = nullptr);

  InferErrorCode terminate();

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
