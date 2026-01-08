/**
 * @file cuda_pinned_host_buffer_impl.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-01-06
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "ai_core/pinned_host_buffer_impl.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ai_core {

/**
 * @brief CUDA implementation of PinnedHostBufferImpl
 *
 * Uses cudaMallocHost for allocation and cudaFreeHost for deallocation.
 * Memory is automatically freed when the object is destroyed.
 */
class CudaPinnedHostBufferImpl : public PinnedHostBufferImpl {
public:
  explicit CudaPinnedHostBufferImpl(size_t sizeBytes)
      : m_ptr(nullptr), m_sizeBytes(sizeBytes) {
    if (sizeBytes > 0) {
      cudaError_t err = cudaMallocHost(&m_ptr, sizeBytes);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                                 cudaGetErrorString(err));
      }
      // Zero-initialize for safety
      std::memset(m_ptr, 0, sizeBytes);
    }
  }

  ~CudaPinnedHostBufferImpl() override {
    if (m_ptr) {
      cudaFreeHost(m_ptr);
      m_ptr = nullptr;
    }
  }

  // Non-copyable
  CudaPinnedHostBufferImpl(const CudaPinnedHostBufferImpl &) = delete;
  CudaPinnedHostBufferImpl &
  operator=(const CudaPinnedHostBufferImpl &) = delete;

  // Movable
  CudaPinnedHostBufferImpl(CudaPinnedHostBufferImpl &&other) noexcept
      : m_ptr(other.m_ptr), m_sizeBytes(other.m_sizeBytes) {
    other.m_ptr = nullptr;
    other.m_sizeBytes = 0;
  }

  CudaPinnedHostBufferImpl &
  operator=(CudaPinnedHostBufferImpl &&other) noexcept {
    if (this != &other) {
      if (m_ptr) {
        cudaFreeHost(m_ptr);
      }
      m_ptr = other.m_ptr;
      m_sizeBytes = other.m_sizeBytes;
      other.m_ptr = nullptr;
      other.m_sizeBytes = 0;
    }
    return *this;
  }

  void *get() const override { return m_ptr; }

  size_t getSizeBytes() const override { return m_sizeBytes; }

private:
  void *m_ptr;
  size_t m_sizeBytes;
};

// ============================================================================
// Static Factory Implementations
// ============================================================================

std::unique_ptr<PinnedHostBufferImpl>
PinnedHostBufferImpl::create(size_t sizeBytes) {
  return std::make_unique<CudaPinnedHostBufferImpl>(sizeBytes);
}

std::unique_ptr<PinnedHostBufferImpl>
PinnedHostBufferImpl::clone(const PinnedHostBufferImpl &other) {
  auto cloned =
      std::make_unique<CudaPinnedHostBufferImpl>(other.getSizeBytes());
  if (other.getSizeBytes() > 0 && other.get() && cloned->get()) {
    std::memcpy(cloned->get(), other.get(), other.getSizeBytes());
  }
  return cloned;
}

} // namespace ai_core