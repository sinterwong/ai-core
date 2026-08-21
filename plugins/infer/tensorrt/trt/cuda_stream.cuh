#ifndef CUDA_UTILS_CUDA_STREAM_CUH
#define CUDA_UTILS_CUDA_STREAM_CUH
#include "cuda_helper.cuh"
namespace ai_core::cuda_utils {
/** Owns a non-blocking CUDA stream. */
class CudaStream {
public:
  enum class Priority { Default, High, Low };

  explicit CudaStream(Priority priority = Priority::Default) {
    unsigned int flags = cudaStreamNonBlocking;

    if (priority == Priority::Default) {
      CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&m_stream, flags));
    } else {
      int least_priority, greatest_priority;
      CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&least_priority,
                                                        &greatest_priority));

      int stream_priority =
          (priority == Priority::High) ? greatest_priority : least_priority;
      CHECK_CUDA_ERROR(
          cudaStreamCreateWithPriority(&m_stream, flags, stream_priority));
    }
  }

  ~CudaStream() {
    if (m_stream) {
      cudaStreamDestroy(m_stream);
      m_stream = nullptr;
    }
  }

  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

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

  cudaStream_t get() const { return m_stream; }

  operator cudaStream_t() const { return m_stream; }

  void synchronize() const {
    if (m_stream) {
      CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
    }
  }

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
