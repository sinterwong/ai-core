#ifndef AI_CORE_ALGO_POSTPROC_HPP
#define AI_CORE_ALGO_POSTPROC_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

/**
 * @brief Standalone postprocessing stage (bypasses the AlgoInference facade).
 *
 * @par Thread safety
 * `process()` and `batchProcess()` may run concurrently after initialization;
 * call-local scratch travels through `TensorData` and `RuntimeContext`.
 * `initialize()` and `terminate()` require exclusive access.
 */
class AlgoPostproc {
public:
  AlgoPostproc(const std::string &module_name);

  ~AlgoPostproc();

  /**
   * @brief Create the plugin and bind validated default parameters.
   */
  InferErrorCode initialize(const AlgoPostprocParams &postproc_params);

  /**
   * @brief Decode one set of model output tensors into an algorithm result.
   *
   * @param postproc_override Per-call replacement for the bound parameters, or
   * `nullptr` to use the defaults from `initialize()`.
   */
  InferErrorCode process(const TensorData &model_output, AlgoOutput &output,
                         std::shared_ptr<RuntimeContext> &runtime_context,
                         const AlgoPostprocParams *postproc_override = nullptr);

  /** Decode batched model output into one result per input. */
  InferErrorCode
  batchProcess(const TensorData &model_output, std::vector<AlgoOutput> &output,
               std::shared_ptr<RuntimeContext> &runtime_context,
               const AlgoPostprocParams *postproc_override = nullptr);

  InferErrorCode terminate();

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
