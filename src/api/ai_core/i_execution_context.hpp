/**
 * @file i_execution_context.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Abstract interface for asynchronous inference execution context
 * @version 0.2
 * @date 2026-01-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_EXECUTION_CONTEXT_HPP
#define AI_CORE_EXECUTION_CONTEXT_HPP

#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <future>
#include <typeinfo>

namespace ai_core::dnn {

/**
 * @brief Type-safe wrapper for backend-specific handles
 *
 * Provides generic access to underlying platform handles without exposing
 * headers like <cuda_runtime.h>, <CL/cl.h>, etc.
 *
 * Examples:
 * - CUDA: cudaStream_t
 * - OpenCL: cl_command_queue
 * - CPU: std::thread::id or custom thread pool handle
 * - NPU: aclrtStream (Ascend), etc.
 */
class BackendHandle {
public:
  BackendHandle() : m_handle(nullptr), m_typeHash(0) {}

  template <typename T>
  explicit BackendHandle(T handle)
      : m_handle(reinterpret_cast<void *>(handle)),
        m_typeHash(typeid(T).hash_code()) {}

  /**
   * @brief Get the handle as a specific native type
   * @throws std::bad_cast if the requested type doesn't match the stored handle
   */
  template <typename T> T as() const {
    if (typeid(T).hash_code() != m_typeHash) {
      throw std::bad_cast();
    }
    return reinterpret_cast<T>(m_handle);
  }

  explicit operator bool() const { return m_handle != nullptr; }
  void *raw() const { return m_handle; }

private:
  void *m_handle;
  size_t m_typeHash;
};

/**
 * @brief Abstract interface for an independent execution context
 *
 * An Execution Context represents a stateful "channel" for inference
 * operations. It encapsulates:
 * - Command Queues (e.g., CUDA Stream, OpenCL Queue)
 * - Temporary Workspaces (Device Memory)
 * - Backend-specific execution states
 * - Optimization profiles (e.g., Captured Graphs)
 *
 * Thread Safety:
 * - An IExecutionContext is NOT thread-safe.
 * - It should be created by an Engine and owned by a single worker thread.
 *
 * Usage Pattern:
 * @code
 * auto context = async_engine->createExecutionContext();
 *
 * // Submit async task
 * auto future = context->inferAsync(inputs, outputs);
 *
 * // Do other work...
 *
 * // Wait for completion
 * context->synchronize();
 * @endcode
 */
class IExecutionContext {
public:
  virtual ~IExecutionContext() = default;

  /**
   * @brief Submit an asynchronous inference task
   *
   * @param inputs Input tensors (Use allocateAcceleratorBuffer for best
   * performance)
   * @param outputs Output tensors
   * @return std::future<InferErrorCode> Future resolving upon completion
   *
   * @note
   * - This call returns immediately (non-blocking).
   * - Data usage: Ensure inputs remain valid until the operation completes
   *   (unless the backend performs an internal copy).
   */
  virtual std::future<InferErrorCode> inferAsync(const TensorData &inputs,
                                                 TensorData &outputs) = 0;

  /**
   * @brief Block until all pending operations in this context complete
   */
  virtual InferErrorCode synchronize() = 0;

  /**
   * @brief Non-blocking check for completion status
   */
  virtual bool isComplete() const = 0;

  /**
   * @brief Get the native backend handle
   * Used for interoperability with other libraries (e.g., OpenCV with CUDA,
   * NPP).
   */
  virtual BackendHandle getHandle() const = 0;

  /**
   * @brief Enable/Disable execution graph capture (Optimization)
   *
   * Generic interface for features like:
   * - CUDA Graphs
   * - OpenVINO Compiled Request caching
   * - Static command buffer recording
   *
   * @param enable true to start capturing/using the graph
   * @return InferErrorCode::SUCCESS if supported and changed
   *
   * @note When enabled, input/output memory addresses typically must remain
   * fixed.
   */
  virtual InferErrorCode setGraphEnabled(bool enable) = 0;

  /**
   * @brief Check if graph/optimization mode is active
   */
  virtual bool isGraphEnabled() const = 0;
};

} // namespace ai_core::dnn

#endif // AI_CORE_EXECUTION_CONTEXT_HPP