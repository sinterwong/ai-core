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

#include <cstddef>
namespace ai_core::trt_utils {

class TrtDeviceBuffer {
public:
  TrtDeviceBuffer();

  explicit TrtDeviceBuffer(size_t sizeBytes);

  TrtDeviceBuffer(const TrtDeviceBuffer &other);

  TrtDeviceBuffer &operator=(const TrtDeviceBuffer &other);

  TrtDeviceBuffer(TrtDeviceBuffer &&other) noexcept;

  TrtDeviceBuffer &operator=(TrtDeviceBuffer &&other) noexcept;

  ~TrtDeviceBuffer();

  void *get() const;

  size_t getSizeBytes() const;

  void release();

  void swap(TrtDeviceBuffer &other) noexcept;

private:
  void *mBuffer_;
  size_t mSizeBytes_;
};
} // namespace ai_core::trt_utils

#endif // __TRT_DEVICE_BUFFER_HPP__