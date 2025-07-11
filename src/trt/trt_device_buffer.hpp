/**
 * @file trt_device_buffer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __TRT_DEVICE_BUFFER_HPP__
#define __TRT_DEVICE_BUFFER_HPP__

#include "trt_utils.hpp"
#include <cuda_runtime_api.h>

namespace ai_core::trt_utils {

class TrtDeviceBuffer {
public:
  TrtDeviceBuffer() : mBuffer_(nullptr), mSizeBytes_(0) {}

  explicit TrtDeviceBuffer(size_t sizeBytes)
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

  TrtDeviceBuffer(const TrtDeviceBuffer &) = delete;
  TrtDeviceBuffer &operator=(const TrtDeviceBuffer &) = delete;

  TrtDeviceBuffer(TrtDeviceBuffer &&other) noexcept
      : mBuffer_(other.mBuffer_), mSizeBytes_(other.mSizeBytes_) {
    other.mBuffer_ = nullptr;
    other.mSizeBytes_ = 0;
  }

  TrtDeviceBuffer &operator=(TrtDeviceBuffer &&other) noexcept {
    if (this != &other) {
      release();
      mBuffer_ = other.mBuffer_;
      mSizeBytes_ = other.mSizeBytes_;
      other.mBuffer_ = nullptr;
      other.mSizeBytes_ = 0;
    }
    return *this;
  }

  ~TrtDeviceBuffer() { release(); }

  void *get() const { return mBuffer_; }

  size_t getSizeBytes() const { return mSizeBytes_; }

  void release() {
    if (mBuffer_) {
      cudaFree(mBuffer_);
      mBuffer_ = nullptr;
      mSizeBytes_ = 0;
    }
  }

private:
  void *mBuffer_;
  size_t mSizeBytes_;
};
} // namespace ai_core::trt_utils

#endif // __TRT_DEVICE_BUFFER_HPP__