#ifndef AI_CORE_EXECUTION_CONTEXT_HPP
#define AI_CORE_EXECUTION_CONTEXT_HPP

#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <future>
#include <typeinfo>

namespace ai_core::dnn {

/**
 * @brief Type-checked wrapper for an optional backend-native handle.
 *
 * Provides generic access to underlying platform handles without exposing
 * vendor headers in the public API.
 *
 * Typical values include a CUDA stream or an OpenCL command queue.
 */
class BackendHandle {
public:
  BackendHandle() : m_handle(nullptr), m_typeHash(0) {}

  template <typename T>
  explicit BackendHandle(T handle)
      : m_handle(reinterpret_cast<void *>(handle)),
        m_typeHash(typeid(T).hash_code()) {}

  /**
   * @brief Return the handle as its original native type.
   * @throws std::bad_cast If `T` differs from the type used at construction.
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
 * @brief Stateful channel for asynchronous inference on one backend.
 *
 * A context owns its command queue, temporary workspace, and backend execution
 * state independently of other contexts created by the same engine.
 *
 * @par Thread safety
 * A context is not thread-safe. One worker must own it and must not submit a
 * new operation while another operation on that context remains pending.
 */
class IExecutionContext {
public:
  virtual ~IExecutionContext() = default;

  /**
   * @brief Submit inference and return before backend work completes.
   *
   * `inputs`, their buffers, and the `outputs` object must remain alive and
   * must not be mutated until the returned future becomes ready or
   * `synchronize()` completes. The backend populates `outputs` before
   * reporting completion.
   */
  virtual std::future<InferErrorCode> inferAsync(const TensorData &inputs,
                                                 TensorData &outputs) = 0;

  /**
   * @brief Block until all operations previously submitted to this context
   * complete.
   */
  virtual InferErrorCode synchronize() = 0;

  /**
   * @brief Report whether this context has no pending backend work.
   */
  virtual bool isComplete() const = 0;

  /**
   * @brief Return the native command-queue handle for interoperability.
   *
   * An empty handle means the backend exposes no interoperable native handle.
   */
  virtual BackendHandle getHandle() const = 0;

  /**
   * @brief Enable or disable the backend's reusable execution graph.
   *
   * Backends that do not support graph execution return an error. When graph
   * execution is enabled, callers must keep input and output addresses stable
   * between submissions unless the backend documents a weaker constraint.
   */
  virtual InferErrorCode setGraphEnabled(bool enable) = 0;

  /**
   * @brief Report whether reusable graph execution is active.
   */
  virtual bool isGraphEnabled() const = 0;
};

} // namespace ai_core::dnn

#endif // AI_CORE_EXECUTION_CONTEXT_HPP
