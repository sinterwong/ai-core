#ifndef AI_CORE_INFER_ASYNC_HPP
#define AI_CORE_INFER_ASYNC_HPP

#include "ai_core/i_execution_context.hpp"
#include "ai_core/i_infer_engine.hpp"

namespace ai_core::dnn {

/**
 * @brief Inference engine that creates independent asynchronous contexts.
 *
 * The engine owns shared model and hardware resources. Each
 * `IExecutionContext` owns the per-execution state used to submit work.
 *
 * @par Thread safety
 * Context creation and accelerator-buffer allocation may be called
 * concurrently. Each returned `IExecutionContext` is not thread-safe and must
 * be owned by one worker at a time.
 */
class IAsyncInferEngine : public IInferEnginePlugin {
public:
  virtual ~IAsyncInferEngine() = default;

  /**
   * @brief Create an independent context that shares this engine's model.
   */
  virtual std::shared_ptr<IExecutionContext> createExecutionContext() = 0;

  /**
   * @brief Allocate host-visible storage optimized for transfers to this
   * backend.
   *
   * The returned `TypedBuffer` owns its storage. Its memory descriptor records
   * whether the backend selected pinned, unified, or ordinary host memory.
   */
  virtual TypedBuffer allocateAcceleratorBuffer(DataType type,
                                                size_t size_bytes) = 0;

  /**
   * @brief Execution context bundled with reusable named input/output buffers.
   */
  struct ContextPackage {
    std::shared_ptr<IExecutionContext> context;
    TensorData inputs;  ///< Reusable backend-optimized input buffers.
    TensorData outputs; ///< Reusable backend-optimized output buffers.
  };

  /**
   * @brief Create a context package suitable for stable-address execution.
   *
   * A backend override may pre-populate the tensor buffers required by graph
   * execution or other static-memory modes.
   */
  virtual ContextPackage createContextPackage() {
    return {createExecutionContext(), {}, {}};
  }

  /**
   * @brief Create `count` mutually independent execution contexts.
   */
  virtual std::vector<std::shared_ptr<IExecutionContext>>
  createContextPool(size_t count) {
    std::vector<std::shared_ptr<IExecutionContext>> pool;
    pool.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      pool.push_back(createExecutionContext());
    }
    return pool;
  }
};

} // namespace ai_core::dnn

#endif // AI_CORE_INFER_ASYNC_HPP
