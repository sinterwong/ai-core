/**
 * @file gpu_buffer_impl.cu
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "gpu_buffer_impl.hpp"
#include <cuda_runtime_api.h>

namespace ai_core {
void GpuBufferImpl::GpuBufferDeleter::operator()(void *ptr) const {
  if (ptr) {
    cudaFree(ptr);
  }
}

GpuBufferImpl::GpuBufferImpl(size_t sizeBytes) : mSizeBytes(sizeBytes) {
  void *ptr = nullptr;
  if (mSizeBytes > 0) {
    cudaMalloc(&ptr, mSizeBytes);
  }
  mManagedBuffer.reset(ptr, GpuBufferDeleter());
  mPtr = ptr;
}

GpuBufferImpl::GpuBufferImpl(void *ptr, size_t sizeBytes, bool manageMemory)
    : mPtr(ptr), mSizeBytes(sizeBytes) {
  if (manageMemory) {
    mManagedBuffer.reset(ptr, GpuBufferDeleter());
  }
}

GpuBufferImpl::~GpuBufferImpl() {}

void *GpuBufferImpl::get() const { return mPtr; }

size_t GpuBufferImpl::getSizeBytes() const { return mSizeBytes; }

} // namespace ai_core