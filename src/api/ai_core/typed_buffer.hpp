/**
 * @file typed_buffer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __TYPED_BUFFER_HPP__
#define __TYPED_BUFFER_HPP__
#include "ai_core/device_buffer_impl.hpp"
#include "ai_core/infer_common_types.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ai_core {

enum class BufferLocation { CPU, GPU_DEVICE };

class TypedBuffer {
public:
  TypedBuffer();
  ~TypedBuffer();

  TypedBuffer(const TypedBuffer &other);
  TypedBuffer &operator=(const TypedBuffer &other);
  TypedBuffer(TypedBuffer &&other) noexcept;
  TypedBuffer &operator=(TypedBuffer &&other) noexcept;

  static TypedBuffer createFromCpu(DataType type,
                                   const std::vector<uint8_t> &data);
  static TypedBuffer createFromCpu(DataType type, std::vector<uint8_t> &&data);

  static TypedBuffer createFromGpu(DataType type, size_t sizeBytes,
                                   int deviceId = 0);
  static TypedBuffer createFromGpu(DataType type, void *devicePtr,
                                   size_t sizeBytes, int deviceId,
                                   bool manageMemory);

  DataType dataType() const noexcept { return mDataType; }
  BufferLocation location() const noexcept { return mLocation; }
  size_t getSizeBytes() const noexcept;
  size_t getElementCount() const noexcept { return mElementCount; }
  int getDeviceId() const noexcept;

  template <typename T> const T *getHostPtr() const;
  template <typename T> T *getHostPtr();

  const void *getRawHostPtr() const;
  void *getRawHostPtr();

  void *getRawDevicePtr() const;

  void setCpuData(DataType type, const std::vector<uint8_t> &data);
  void setCpuData(DataType type, std::vector<uint8_t> &&data);

  void setGpuDataReference(DataType type, void *ptr, size_t sizeBytes,
                           int devId = 0);

  void resize(size_t newElementCount);

  void clear();

  static size_t getElementSize(DataType type) noexcept;

private:
  TypedBuffer(DataType type, const std::vector<uint8_t> &cpuData);
  TypedBuffer(DataType type, std::vector<uint8_t> &&cpuData);
  TypedBuffer(DataType type, size_t bufferSizeBytes, int deviceId);
  TypedBuffer(DataType type, void *devicePtr, size_t bufferSizeBytes,
              int deviceId, bool manageMemory);

  void reset();

  DataType mDataType{DataType::FLOAT32};
  BufferLocation mLocation{BufferLocation::CPU};

  std::vector<uint8_t> mCpuData;

  std::unique_ptr<DeviceBufferImpl> mGpuImpl;
  size_t mDeviceBufferSizeBytes{0};
  int mDeviceId{0};

  size_t mElementCount{0};
};

template <typename T> const T *TypedBuffer::getHostPtr() const {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  if (sizeof(T) != getElementSize(mDataType) && !mCpuData.empty()) {
    throw std::runtime_error("Mismatched type size for host data access.");
  }
  return reinterpret_cast<const T *>(mCpuData.data());
}

template <typename T> T *TypedBuffer::getHostPtr() {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  if (sizeof(T) != getElementSize(mDataType) && !mCpuData.empty()) {
    throw std::runtime_error("Mismatched type size for host data access.");
  }
  return reinterpret_cast<T *>(mCpuData.data());
}

} // namespace ai_core

#endif
