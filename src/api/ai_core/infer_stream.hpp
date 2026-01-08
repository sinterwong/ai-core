/**
 * @file infer_stream.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Abstract interface for asynchronous inference stream
 * @version 0.1
 * @date 2025-01-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFER_STREAM_HPP
#define AI_CORE_INFER_STREAM_HPP

#include "ai_core/infer_error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <future>
#include <stdexcept>
#include <typeinfo>

namespace ai_core::dnn {

/**
 * @brief Type-safe handle wrapper for native stream handles
 *
 * Provides type-safe access to underlying platform-specific handles
 * (e.g., cudaStream_t for CUDA) without exposing platform headers.
 */
class StreamHandle {
public:
  StreamHandle() : m_handle(nullptr), m_typeHash(0) {}

  template <typename T>
  explicit StreamHandle(T handle)
      : m_handle(reinterpret_cast<void *>(handle)),
        m_typeHash(typeid(T).hash_code()) {}

  /**
   * @brief Get the handle as a specific type
   * @tparam T The expected handle type (e.g., cudaStream_t)
   * @return The handle cast to the specified type
   * @throws std::bad_cast if the type doesn't match
   */
  template <typename T> T as() const {
    if (typeid(T).hash_code() != m_typeHash) {
      throw std::bad_cast();
    }
    return reinterpret_cast<T>(m_handle);
  }

  /**
   * @brief Check if the handle is valid
   */
  explicit operator bool() const { return m_handle != nullptr; }

  /**
   * @brief Get raw pointer (unsafe, use with caution)
   */
  void *raw() const { return m_handle; }

private:
  void *m_handle;
  size_t m_typeHash;
};

/**
 * @brief Abstract interface for an asynchronous inference execution stream
 *
 * Each IInferStream instance represents an independent execution context with:
 * - Its own CUDA stream (for CUDA-based engines)
 * - Its own device memory buffers
 * - Its own execution context
 * - Its own CUDA Graph instance (if enabled)
 *
 * Thread Safety:
 * - Each thread should own its own IInferStream instance
 * - Do NOT share IInferStream instances across threads
 *
 * Usage Pattern:
 * @code
 * auto stream = asyncEngine->createStream();
 *
 * // Option 1: Fire-and-forget with future
 * auto future = stream->inferAsync(inputs, outputs);
 * // ... do other work ...
 * auto status = future.get();  // blocks until complete
 *
 * // Option 2: Explicit synchronization
 * stream->inferAsync(inputs, outputs);
 * // ... do other work ...
 * stream->synchronize();
 * @endcode
 */
class IInferStream {
public:
  virtual ~IInferStream() = default;

  /**
   * @brief Asynchronously submit an inference task
   *
   * The function returns immediately after queuing the work.
   * Use the returned future or synchronize() to wait for completion.
   *
   * @param inputs Input tensors (Pinned Memory recommended for best
   * performance)
   * @param outputs Output tensors (will be populated after synchronization)
   * @return std::future<InferErrorCode> Future that resolves when inference
   * completes
   *
   * @note For optimal performance:
   *  - Use Pinned Memory (cudaMallocHost) for inputs/outputs
   *  - Pre-allocate output buffers to avoid synchronization
   *  - Consider using multiple streams for pipelining
   */
  virtual std::future<InferErrorCode> inferAsync(const TensorData &inputs,
                                                 TensorData &outputs) = 0;

  /**
   * @brief Wait for all pending operations on this stream to complete
   *
   * This is a blocking call that returns only when all previously
   * submitted operations have finished.
   *
   * @return InferErrorCode::SUCCESS if all operations completed successfully
   */
  virtual InferErrorCode synchronize() = 0;

  /**
   * @brief Query if all operations on this stream have completed
   *
   * Non-blocking check for stream completion status.
   *
   * @return true if all operations are complete, false if work is pending
   */
  virtual bool isComplete() const = 0;

  /**
   * @brief Get the native stream handle for interop with other CUDA libraries
   *
   * @return StreamHandle Type-safe wrapper around the native handle
   *
   * @note Use this for advanced scenarios like:
   *  - Synchronizing with other CUDA libraries
   *  - Custom kernel launches on the same stream
   *  - Recording CUDA events
   */
  virtual StreamHandle getHandle() const = 0;

  /**
   * @brief Enable or disable CUDA Graph capture for this stream
   *
   * When enabled, the next inference call will capture a CUDA Graph,
   * and subsequent calls will replay the graph for improved performance.
   *
   * @param enable true to enable, false to disable
   * @return InferErrorCode::SUCCESS on success
   *
   * @note CUDA Graph provides significant speedup for:
   *  - Small batch sizes
   *  - Models with many small kernels
   *  - Scenarios with fixed input shapes
   *
   * @warning CUDA Graph requires fixed memory addresses. Do not reallocate
   *          input/output buffers after graph capture.
   */
  virtual InferErrorCode setGraphEnabled(bool enable) = 0;

  /**
   * @brief Check if CUDA Graph is currently enabled
   */
  virtual bool isGraphEnabled() const = 0;
};

} // namespace ai_core::dnn

#endif // AI_CORE_INFER_STREAM_HPP