#ifndef AI_CORE_INFER_ASYNC_HPP
#define AI_CORE_INFER_ASYNC_HPP

#include "ai_core/i_execution_context.hpp"
#include "ai_core/i_infer_engine.hpp"

namespace ai_core::dnn {

/**
 * @brief Extended inference engine interface with async capabilities
 *
 * This interface represents a "Heavy" resource factory. It manages weights,
 * model definitions, and hardware initialization. It spawns lightweight
 * "IExecutionContext" instances for actual inference.
 *
 * @par Thread safety
 * The engine (weights/factory) is shared and its context-creation methods are
 * safe to call concurrently. Each IExecutionContext it hands out is NOT
 * thread-safe and must be owned by a single worker thread — that is how
 * multiple threads achieve parallel inference against one engine.
 */
class IAsyncInferEngine : public IInferEnginePlugin {
public:
  virtual ~IAsyncInferEngine() = default;

  /**
   * @brief Create an independent execution context (Thread-Local)
   *
   * Creates a lightweight context for submitting tasks.
   * This is equivalent to:
   * - Creating a cudaStream_t (CUDA)
   * - Creating an InferRequest (OpenVINO)
   * - Creating a Command Queue (OpenCL)
   *
   * @return std::shared_ptr<IExecutionContext> New context
   */
  virtual std::shared_ptr<IExecutionContext> createExecutionContext() = 0;

  /**
   * @brief Allocate host memory optimized for the specific accelerator
   *
   * Allocates memory on the CPU that is accessible or optimized for
   * the backend device.
   *
   * Characteristics based on backend:
   * - CUDA/HIP: Returns Pinned Memory (Page-locked).
   * - Integrated GPU/NPU: May return Shared Memory (Zero-Copy).
   * - CPU: Returns aligned memory for AVX/NEON.
   *
   * @param type Data type
   * @param sizeBytes Size in bytes
   * @return TypedBuffer RAII-managed buffer
   */
  virtual TypedBuffer allocateAcceleratorBuffer(DataType type,
                                                size_t size_bytes) = 0;

  /**
   * @brief Structure holding a context and its pre-allocated resources
   */
  struct ContextPackage {
    std::shared_ptr<IExecutionContext> context;
    TensorData inputs;  ///< Pre-allocated optimized input buffers
    TensorData outputs; ///< Pre-allocated optimized output buffers
  };

  /**
   * @brief Create a fully initialized context with buffers
   *
   * Helper for Zero-Copy or Static-Memory scenarios (e.g., CUDA Graph),
   * where memory addresses should not change between runs.
   */
  virtual ContextPackage createContextPackage() {
    return {createExecutionContext(), {}, {}};
  }

  /**
   * @brief Create a pool of contexts for pipelined execution
   *
   * Useful for maximizing hardware utilization by overlapping
   * data transfers and computation across multiple contexts.
   *
   * @param count Number of contexts
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