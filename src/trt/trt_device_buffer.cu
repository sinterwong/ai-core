/**
 * @file trt_device_buffer.cu
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "trt_device_buffer.hpp"
#include "trt_utils.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>

namespace ai_core::trt_utils {

TrtDeviceBuffer::TrtDeviceBuffer() : mBuffer_(nullptr), mSizeBytes_(0) {}

TrtDeviceBuffer::TrtDeviceBuffer(size_t sizeBytes)
    : mBuffer_(nullptr), mSizeBytes_(0) {
  if (sizeBytes > 0) {
    CHECK_CUDA(cudaMalloc(&mBuffer_, sizeBytes));
    if (mBuffer_) {
      mSizeBytes_ = sizeBytes;
    } else {
      throw std::runtime_error(
          "TrtDeviceBuffer: Failed to allocate CUDA memory.");
    }
  }
}

TrtDeviceBuffer::TrtDeviceBuffer(const TrtDeviceBuffer &other)
    : mBuffer_(nullptr), mSizeBytes_(0) {
  if (other.mSizeBytes_ > 0 && other.mBuffer_ != nullptr) {
    mSizeBytes_ = other.mSizeBytes_;
    CHECK_CUDA(cudaMalloc(&mBuffer_, mSizeBytes_));
    CHECK_CUDA(cudaMemcpy(mBuffer_, other.mBuffer_, mSizeBytes_,
                          cudaMemcpyDeviceToDevice));
  }
}

TrtDeviceBuffer &TrtDeviceBuffer::operator=(const TrtDeviceBuffer &other) {
  if (this != &other) {
    // Create a temporary copy, then swap with it.
    // This is exception-safe and handles self-assignment implicitly.
    TrtDeviceBuffer temp(other);
    swap(temp);
  }
  return *this;
}

TrtDeviceBuffer::TrtDeviceBuffer(TrtDeviceBuffer &&other) noexcept
    : mBuffer_(other.mBuffer_), mSizeBytes_(other.mSizeBytes_) {
  other.mBuffer_ = nullptr;
  other.mSizeBytes_ = 0;
}

TrtDeviceBuffer &TrtDeviceBuffer::operator=(TrtDeviceBuffer &&other) noexcept {
  if (this != &other) {
    release();
    mBuffer_ = other.mBuffer_;
    mSizeBytes_ = other.mSizeBytes_;
    other.mBuffer_ = nullptr;
    other.mSizeBytes_ = 0;
  }
  return *this;
}

void TrtDeviceBuffer::swap(TrtDeviceBuffer &other) noexcept {
  std::swap(mBuffer_, other.mBuffer_);
  std::swap(mSizeBytes_, other.mSizeBytes_);
}

TrtDeviceBuffer::~TrtDeviceBuffer() { release(); }

void *TrtDeviceBuffer::get() const { return mBuffer_; }

size_t TrtDeviceBuffer::getSizeBytes() const { return mSizeBytes_; }

void TrtDeviceBuffer::release() {
  if (mBuffer_) {
    cudaFree(mBuffer_);
    mBuffer_ = nullptr;
    mSizeBytes_ = 0;
  }
}
} // namespace ai_core::trt_utils
