#ifndef AI_CORE_ALGO_INFER_BASE_HPP
#define AI_CORE_ALGO_INFER_BASE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/common_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/infer_async.hpp"
#include "ai_core/infer_config.hpp"
#include <memory>

namespace ai_core::dnn {

/**
 * @brief Three-stage inference facade (preprocess -> infer -> postprocess).
 *
 * @par Thread safety
 * `infer()` and `batchInfer()` may run concurrently on one initialized
 * instance. Each call owns its pipeline scratch, and every registered stage
 * must honor its plugin interface's reentrancy contract. `initialize()` and
 * `terminate()` require exclusive access to the instance.
 */
class AlgoInference {
public:
  AlgoInference(const AlgoModuleTypes &module_types,
                const AlgoInferParams &infer_params);

  ~AlgoInference();

  /**
   * @brief Create the three plugins and bind their default parameters.
   *
   * A successful call is required before inference. Per-call overrides replace
   * the bound preprocessor or postprocessor parameters for that call only.
   */
  InferErrorCode initialize(const AlgoPreprocParams &preproc_params,
                            const AlgoPostprocParams &postproc_params);

  /**
   * @brief Run one input through the configured three-stage pipeline.
   *
   * Pass `nullptr` for an override to use the parameters bound by
   * `initialize()`. The input's non-owning data must remain valid until this
   * synchronous call returns.
   */
  InferErrorCode infer(const AlgoInput &input, AlgoOutput &output,
                       const AlgoPreprocParams *preproc_override = nullptr,
                       const AlgoPostprocParams *postproc_override = nullptr);

  /**
   * @brief Run a batch and write results in the order supplied by the caller.
   *
   * Override semantics are identical to `infer()`.
   */
  InferErrorCode
  batchInfer(const std::vector<AlgoInput> &inputs,
             std::vector<AlgoOutput> &outputs,
             const AlgoPreprocParams *preproc_override = nullptr,
             const AlgoPostprocParams *postproc_override = nullptr);

  InferErrorCode terminate();

  /** Returns an empty value before successful initialization. */
  const ModelInfo &getModelInfo() const noexcept;

  /** Returns the stage names copied at construction. */
  const AlgoModuleTypes &getModuleTypes() const noexcept;

  /**
   * @brief Return the asynchronous engine interface when the backend provides
   * it.
   *
   * The front door to the async infrastructure (execution contexts, pinned
   * buffers, and backend graph execution). Call after `initialize()`; a null
   * result means the selected backend exposes only synchronous inference.
   */
  std::shared_ptr<IAsyncInferEngine> getAsyncEngine() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
