#ifndef CUDA_UTILS_CUDA_STREAM_CUH
#define CUDA_UTILS_CUDA_STREAM_CUH
#include "cuda_helper.cuh"
namespace ai_core::cuda_utils {
class CudaStream {
public:
  enum class Priority { Default, High, Low };

  explicit CudaStream(Priority priority = Priority::Default) {
    unsigned int flags = cudaStreamNonBlocking;

    if (priority == Priority::Default) {
      CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&m_stream, flags));
    } else {
      int leastPriority, greatestPriority;
      CHECK_CUDA_ERROR(
          cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

      int streamPriority =
          (priority == Priority::High) ? greatestPriority : leastPriority;
      CHECK_CUDA_ERROR(
          cudaStreamCreateWithPriority(&m_stream, flags, streamPriority));
    }
  }

  ~CudaStream() {
    if (m_stream) {
      cudaStreamDestroy(m_stream);
      m_stream = nullptr;
    }
  }

  // Non-copyable
  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  // Movable
  CudaStream(CudaStream &&other) noexcept : m_stream(other.m_stream) {
    other.m_stream = nullptr;
  }

  CudaStream &operator=(CudaStream &&other) noexcept {
    if (this != &other) {
      if (m_stream) {
        cudaStreamDestroy(m_stream);
      }
      m_stream = other.m_stream;
      other.m_stream = nullptr;
    }
    return *this;
  }

  /// Get the underlying CUDA stream handle
  cudaStream_t get() const { return m_stream; }

  /// Implicit conversion to cudaStream_t for convenience
  operator cudaStream_t() const { return m_stream; }

  /// Synchronize this stream (wait for all operations to complete)
  void synchronize() const {
    if (m_stream) {
      CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
    }
  }

  /// Check if all operations on this stream have completed
  bool isComplete() const {
    if (!m_stream)
      return true;
    cudaError_t status = cudaStreamQuery(m_stream);
    return status == cudaSuccess;
  }

private:
  cudaStream_t m_stream = nullptr;
};
} // namespace ai_core::cuda_utils
#endif