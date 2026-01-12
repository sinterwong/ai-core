/**
 * @file cuda_device_buffer_impl.cu
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-01-12
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "ai_core/accelerator_buffer_impl.hpp"
#include "cuda_helper.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ai_core {

class CudaAcceleratorBuffer : public AcceleratorBufferImpl {
public:
  CudaAcceleratorBuffer(size_t size_bytes, AcceleratorMemoryType type)
      : m_sizeBytes(size_bytes), m_type(type), m_ownsMemory(true) {

    if (m_sizeBytes == 0)
      return;

    if (m_type == AcceleratorMemoryType::Device) {
      CHECK_CUDA_ERROR(cudaMalloc(&m_ptr, m_sizeBytes));
    } else if (m_type == AcceleratorMemoryType::HostPinned) {
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_sizeBytes));
      // Optional: Zero-init pinned memory implies generic CPU usage
      std::memset(m_ptr, 0, m_sizeBytes);
    }
  }

  CudaAcceleratorBuffer(void *ptr, size_t size_bytes, AcceleratorMemoryType type,
                        bool manage)
      : m_ptr(ptr), m_sizeBytes(size_bytes), m_type(type), m_ownsMemory(manage) {}

  ~CudaAcceleratorBuffer() override {
    if (m_ptr && m_ownsMemory) {
      if (m_type == AcceleratorMemoryType::Device) {
        cudaFree(m_ptr);
      } else {
        cudaFreeHost(m_ptr);
      }
    }
  }

  // Disable Copy, Enable Move
  CudaAcceleratorBuffer(const CudaAcceleratorBuffer &) = delete;
  CudaAcceleratorBuffer &operator=(const CudaAcceleratorBuffer &) = delete;

  // Clone constructor helper
  CudaAcceleratorBuffer(const CudaAcceleratorBuffer &other, bool)
      : m_sizeBytes(other.m_sizeBytes), m_type(other.m_type), m_ownsMemory(true) {

    if (m_sizeBytes == 0)
      return;

    // Allocate
    if (m_type == AcceleratorMemoryType::Device) {
      CHECK_CUDA_ERROR(cudaMalloc(&m_ptr, m_sizeBytes));
      // Copy (Device to Device)
      CHECK_CUDA_ERROR(
          cudaMemcpy(m_ptr, other.m_ptr, m_sizeBytes, cudaMemcpyDeviceToDevice));
    } else {
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_sizeBytes));
      // Copy (Host to Host)
      std::memcpy(m_ptr, other.m_ptr, m_sizeBytes);
    }
  }

  void *get() const override { return m_ptr; }
  size_t getSizeBytes() const override { return m_sizeBytes; }
  AcceleratorMemoryType getType() const override { return m_type; }

private:
  void *m_ptr{nullptr};
  size_t m_sizeBytes{0};
  AcceleratorMemoryType m_type;
  bool m_ownsMemory;
};

// ============================================================================
// Factory Implementation
// ============================================================================

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::create(size_t size_bytes, AcceleratorMemoryType type) {
  return std::make_unique<CudaAcceleratorBuffer>(size_bytes, type);
}

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::createReference(void *ptr, size_t size_bytes,
                                       AcceleratorMemoryType type,
                                       bool manage_memory) {
  return std::make_unique<CudaAcceleratorBuffer>(ptr, size_bytes, type,
                                                 manage_memory);
}

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::clone(const AcceleratorBufferImpl &other) {
  // Dynamic cast ensures we are cloning a compatible CUDA buffer
  const auto *cuda_impl = dynamic_cast<const CudaAcceleratorBuffer *>(&other);
  if (!cuda_impl) {
    throw std::runtime_error(
        "Cannot clone incompatible accelerator buffer type.");
  }
  // Invoke private clone constructor
  return std::make_unique<CudaAcceleratorBuffer>(*cuda_impl, true);
}

} // namespace ai_core
