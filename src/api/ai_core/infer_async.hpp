/**
 * @file infer_async.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Async-capable inference engine interface extension
 * @version 0.1
 * @date 2025-01-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFER_ASYNC_HPP
#define AI_CORE_INFER_ASYNC_HPP

#include "ai_core/infer_base.hpp"
#include "ai_core/infer_stream.hpp"

namespace ai_core::dnn {

/**
 * @brief Extended inference engine interface with async capabilities
 *
 * This interface extends IInferEnginePlugin to support asynchronous
 * operations, primarily designed for GPU-accelerated engines like TensorRT.
 *
 * Design Philosophy:
 * - Maintains backward compatibility with IInferEnginePlugin
 * - Adds async capabilities through composition (IInferStream)
 * - Allows runtime capability detection via dynamic_pointer_cast
 *
 * Usage Pattern:
 * @code
 * auto engine = factory.createEngine("model.engine");
 * engine->initialize();
 *
 * // Check for async capability
 * if (auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine))
 * {
 *     // High-performance async path
 *     auto stream = asyncEngine->createStream();
 *     stream->setGraphEnabled(true);  // Enable CUDA Graph
 *     auto pinnedInput = asyncEngine->allocatePinnedHostBuffer(
 *         DataType::FLOAT32, inputSize);
 *     // ... async inference with stream
 * } else {
 *     // Fallback to sync path (ONNX Runtime, NCNN, etc.)
 *     engine->infer(inputs, outputs);
 * }
 * @endcode
 *
 * Thread Safety:
 * - createStream() is thread-safe and can be called from multiple threads
 * - Each returned IInferStream should be used by only one thread
 * - The base class infer() method remains thread-safe (with internal locking)
 */
class IAsyncInferEngine : public IInferEnginePlugin {
public:
  virtual ~IAsyncInferEngine() = default;

  /**
   * @brief Create an independent inference stream
   *
   * Each stream has its own:
   * - CUDA stream handle
   * - Device memory buffers (input/output)
   * - Execution context
   * - CUDA Graph (if enabled)
   *
   * @return std::shared_ptr<IInferStream> A new inference stream
   *
   * @note Call this once per worker thread, then reuse the stream
   *       for all inferences on that thread.
   *
   * @throws std::runtime_error if engine is not initialized
   */
  virtual std::shared_ptr<IInferStream> createStream() = 0;

  /**
   * @brief Allocate page-locked (pinned) host memory
   *
   * Pinned memory enables:
   * - Asynchronous Host<->Device transfers
   * - Higher transfer bandwidth via DMA
   * - Overlap of data transfer with computation
   *
   * @param type Data type for the buffer
   * @param sizeBytes Size in bytes to allocate
   * @return TypedBuffer A buffer backed by pinned memory
   *
   * @note The returned TypedBuffer:
   *  - Has location = BufferLocation::CPU
   *  - Has memoryType = BufferMemoryType::Pinned
   *  - Is managed by RAII (automatically freed)
   *
   * @warning Pinned memory allocation is expensive. Allocate once and reuse.
   *
   * @throws std::runtime_error if allocation fails
   */
  virtual TypedBuffer allocatePinnedHostBuffer(DataType type,
                                               size_t sizeBytes) = 0;

  /**
   * @brief Pre-allocated stream context for optimal pipelining
   *
   * Contains a stream with pre-allocated pinned memory buffers
   * for zero-copy data transfer patterns.
   */
  struct StreamContext {
    std::shared_ptr<IInferStream> stream;
    TensorData pinnedInputs;  ///< Pre-allocated pinned input buffers
    TensorData pinnedOutputs; ///< Pre-allocated pinned output buffers
  };

  /**
   * @brief Create a stream with pre-allocated pinned buffers
   *
   * This is the recommended method for maximum performance when:
   * - Input/output shapes are known in advance
   * - You want to avoid per-inference allocation overhead
   *
   * @return StreamContext Complete context ready for inference
   *
   * Default implementation creates stream without pre-allocation.
   * Derived classes should override for optimized buffer management.
   */
  virtual StreamContext createStreamContext() {
    // Default: just create stream, no pre-allocation
    return {createStream(), {}, {}};
  }

  /**
   * @brief Create multiple streams for pipelining
   *
   * Pipeline pattern (N streams rotating):
   * @code
   * Stream0: [H2D] [Compute] [D2H]
   * Stream1:       [H2D] [Compute] [D2H]
   * Stream2:             [H2D] [Compute] [D2H]
   *                  ↑ Overlapped execution hides latency
   * @endcode
   *
   * @param count Number of streams to create
   * @return std::vector<std::shared_ptr<IInferStream>> Stream pool
   */
  virtual std::vector<std::shared_ptr<IInferStream>>
  createStreamPool(size_t count) {
    std::vector<std::shared_ptr<IInferStream>> pool;
    pool.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      pool.push_back(createStream());
    }
    return pool;
  }
};

} // namespace ai_core::dnn

#endif // AI_CORE_INFER_ASYNC_HPP