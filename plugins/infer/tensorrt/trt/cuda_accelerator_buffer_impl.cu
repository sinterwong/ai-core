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

#include "ai_core/buffer_storage.hpp"
#include "ai_core/typed_buffer.hpp"
#include "cuda_buffer_storage.hpp"
#include "cuda_helper.cuh"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ai_core {

class CudaAcceleratorBuffer : public IBufferStorage {
public:
  CudaAcceleratorBuffer(size_t size_bytes, MemoryKind type, int device_id = 0)
      : m_sizeBytes(size_bytes), m_type(type), m_deviceId(device_id),
        m_ownsMemory(true) {

    if (m_sizeBytes == 0)
      return;

    if (m_type == MemoryKind::Device) {
      CHECK_CUDA_ERROR(cudaMalloc(&m_ptr, m_sizeBytes));
    } else if (m_type == MemoryKind::HostPinned) {
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_sizeBytes));
      // Optional: Zero-init pinned memory implies generic CPU usage
      std::memset(m_ptr, 0, m_sizeBytes);
    }
  }

  CudaAcceleratorBuffer(void *ptr, size_t size_bytes,
                        MemoryKind type, bool manage, int device_id = 0)
      : m_ptr(ptr), m_sizeBytes(size_bytes), m_type(type),
        m_deviceId(device_id), m_ownsMemory(manage) {}

  ~CudaAcceleratorBuffer() override {
    if (m_ptr && m_ownsMemory) {
      if (m_type == MemoryKind::Device) {
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
      : m_sizeBytes(other.m_sizeBytes), m_type(other.m_type),
        m_deviceId(other.m_deviceId),
        m_ownsMemory(true) {

    if (m_sizeBytes == 0)
      return;

    // Allocate
    if (m_type == MemoryKind::Device) {
      CHECK_CUDA_ERROR(cudaMalloc(&m_ptr, m_sizeBytes));
      // Copy (Device to Device)
      CHECK_CUDA_ERROR(cudaMemcpy(m_ptr, other.m_ptr, m_sizeBytes,
                                  cudaMemcpyDeviceToDevice));
    } else {
      CHECK_CUDA_ERROR(cudaMallocHost(&m_ptr, m_sizeBytes));
      // Copy (Host to Host)
      std::memcpy(m_ptr, other.m_ptr, m_sizeBytes);
    }
  }

  void *data() noexcept override { return m_ptr; }
  const void *data() const noexcept override { return m_ptr; }
  size_t sizeBytes() const noexcept override { return m_sizeBytes; }
  MemoryDescriptor descriptor() const noexcept override {
    return {m_type, "cuda", m_deviceId};
  }
  std::unique_ptr<IBufferStorage> clone() const override {
    return std::make_unique<CudaAcceleratorBuffer>(*this, true);
  }
  std::unique_ptr<IBufferStorage> allocate(size_t bytes) const override {
    return std::make_unique<CudaAcceleratorBuffer>(bytes, m_type, m_deviceId);
  }

private:
  void *m_ptr{nullptr};
  size_t m_sizeBytes{0};
  MemoryKind m_type;
  int m_deviceId{0};
  bool m_ownsMemory;
};

} // namespace ai_core

namespace ai_core::dnn::gpu {

TypedBuffer allocateCudaDeviceBuffer(DataType type, size_t size_bytes,
                                     int device_id) {
  return TypedBuffer::fromStorage(
      type, std::make_unique<CudaAcceleratorBuffer>(
                size_bytes, MemoryKind::Device, device_id));
}

TypedBuffer allocateCudaPinnedBuffer(DataType type, size_t size_bytes,
                                     int device_id) {
  return TypedBuffer::fromStorage(
      type, std::make_unique<CudaAcceleratorBuffer>(
                size_bytes, MemoryKind::HostPinned, device_id));
}

} // namespace ai_core::dnn::gpu
