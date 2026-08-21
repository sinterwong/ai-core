#ifndef AI_CORE_INFER_BASE_HPP
#define AI_CORE_INFER_BASE_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {
/**
 * @brief Synchronous inference engine plugin interface.
 *
 * @par Thread safety
 * Implementations must accept concurrent `infer()` calls, whether by
 * serializing internally or using independent backend contexts.
 * `initialize()` and `terminate()` require exclusive access. Use
 * `IAsyncInferEngine` when callers need explicit execution contexts.
 */
class IInferEnginePlugin {
public:
  IInferEnginePlugin() = default;

  virtual ~IInferEnginePlugin() {}

  /** Initialize model and backend resources before the first inference call. */
  virtual InferErrorCode initialize() = 0;

  /** Run synchronous inference; `outputs` is complete when the call returns. */
  virtual InferErrorCode infer(const TensorData &inputs,
                               TensorData &outputs) = 0;

  /** Release backend resources after all inference calls have completed. */
  virtual InferErrorCode terminate() = 0;

  /** Return metadata owned by the initialized engine. */
  virtual const ModelInfo &getModelInfo() = 0;

  virtual void prettyPrintModelInfos();

protected:
  std::shared_ptr<ModelInfo> m_modelInfo;
};
} // namespace ai_core::dnn
#endif
