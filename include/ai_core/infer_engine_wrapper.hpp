#ifndef AI_CORE_ALGO_INFER_ENGINE_HPP
#define AI_CORE_ALGO_INFER_ENGINE_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/infer_async.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

/**
 * @brief Thin wrapper around a backend inference engine plugin.
 *
 * @par Thread safety
 * `infer()` may run concurrently on one initialized instance; the selected
 * plugin implements the required synchronization. `initialize()` and
 * `terminate()` require exclusive access.
 */
class AlgoInferEngine {
public:
  AlgoInferEngine(const std::string &module_name,
                  const AlgoInferParams &infer_params);

  ~AlgoInferEngine();

  /** Create and initialize the selected backend plugin. */
  InferErrorCode initialize();

  /** Run synchronous inference and fully populate `model_output`. */
  InferErrorCode infer(const TensorData &model_input, TensorData &model_output);

  InferErrorCode terminate();

  /** Return an empty value until the engine has been initialized. */
  const ModelInfo &getModelInfo() const noexcept;

  /**
   * @brief Return the asynchronous interface when the plugin implements it.
   *
   * This is the supported way to reach the async infrastructure (execution
   * contexts, optimized buffers, and backend graph execution. Call after
   * `initialize()`; a null result means the plugin is synchronous only.
   */
  std::shared_ptr<IAsyncInferEngine> getAsyncEngine() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
