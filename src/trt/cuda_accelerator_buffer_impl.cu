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
  CudaAcceleratorBuffer(size_t sizeBytes, AcceleratorMemoryType type)
      : mSizeBytes(sizeBytes), mType(type), mOwnsMemory(true) {

    if (mSizeBytes == 0)
      return;

    if (mType == AcceleratorMemoryType::Device) {
      CHECK_CUDA_ERROR(cudaMalloc(&mPtr, mSizeBytes));
    } else if (mType == AcceleratorMemoryType::HostPinned) {
      CHECK_CUDA_ERROR(cudaMallocHost(&mPtr, mSizeBytes));
      // Optional: Zero-init pinned memory implies generic CPU usage
      std::memset(mPtr, 0, mSizeBytes);
    }
  }

  CudaAcceleratorBuffer(void *ptr, size_t sizeBytes, AcceleratorMemoryType type,
                        bool manage)
      : mPtr(ptr), mSizeBytes(sizeBytes), mType(type), mOwnsMemory(manage) {}

  ~CudaAcceleratorBuffer() override {
    if (mPtr && mOwnsMemory) {
      if (mType == AcceleratorMemoryType::Device) {
        cudaFree(mPtr);
      } else {
        cudaFreeHost(mPtr);
      }
    }
  }

  // Disable Copy, Enable Move
  CudaAcceleratorBuffer(const CudaAcceleratorBuffer &) = delete;
  CudaAcceleratorBuffer &operator=(const CudaAcceleratorBuffer &) = delete;

  // Clone constructor helper
  CudaAcceleratorBuffer(const CudaAcceleratorBuffer &other, bool)
      : mSizeBytes(other.mSizeBytes), mType(other.mType), mOwnsMemory(true) {

    if (mSizeBytes == 0)
      return;

    // Allocate
    if (mType == AcceleratorMemoryType::Device) {
      CHECK_CUDA_ERROR(cudaMalloc(&mPtr, mSizeBytes));
      // Copy (Device to Device)
      CHECK_CUDA_ERROR(
          cudaMemcpy(mPtr, other.mPtr, mSizeBytes, cudaMemcpyDeviceToDevice));
    } else {
      CHECK_CUDA_ERROR(cudaMallocHost(&mPtr, mSizeBytes));
      // Copy (Host to Host)
      std::memcpy(mPtr, other.mPtr, mSizeBytes);
    }
  }

  void *get() const override { return mPtr; }
  size_t getSizeBytes() const override { return mSizeBytes; }
  AcceleratorMemoryType getType() const override { return mType; }

private:
  void *mPtr{nullptr};
  size_t mSizeBytes{0};
  AcceleratorMemoryType mType;
  bool mOwnsMemory;
};

// ============================================================================
// Factory Implementation
// ============================================================================

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::create(size_t sizeBytes, AcceleratorMemoryType type) {
  return std::make_unique<CudaAcceleratorBuffer>(sizeBytes, type);
}

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::createReference(void *ptr, size_t sizeBytes,
                                       AcceleratorMemoryType type,
                                       bool manageMemory) {
  return std::make_unique<CudaAcceleratorBuffer>(ptr, sizeBytes, type,
                                                 manageMemory);
}

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::clone(const AcceleratorBufferImpl &other) {
  // Dynamic cast ensures we are cloning a compatible CUDA buffer
  const auto *cudaImpl = dynamic_cast<const CudaAcceleratorBuffer *>(&other);
  if (!cudaImpl) {
    throw std::runtime_error(
        "Cannot clone incompatible accelerator buffer type.");
  }
  // Invoke private clone constructor
  return std::make_unique<CudaAcceleratorBuffer>(*cudaImpl, true);
}

} // namespace ai_core
