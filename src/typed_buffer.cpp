/**
 * @file typed_buffer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/typed_buffer.hpp"
#include <logger.hpp>

namespace ai_core {

TypedBuffer::TypedBuffer(const TypedBuffer &other)
    : mDataType(other.mDataType), mLocation(other.mLocation),
      mCpuData(other.mCpuData), mDevicePtr(other.mDevicePtr),
      mDeviceBufferSizeBytes(other.mDeviceBufferSizeBytes),
      mDeviceId(other.mDeviceId), mElementCount(other.mElementCount) {}

TypedBuffer &TypedBuffer::operator=(const TypedBuffer &other) {
  if (this != &other) {
    mDataType = other.mDataType;
    mLocation = other.mLocation;
    mCpuData = other.mCpuData;
    mDevicePtr = other.mDevicePtr;
    mDeviceBufferSizeBytes = other.mDeviceBufferSizeBytes;
    mDeviceId = other.mDeviceId;
    mElementCount = other.mElementCount;
  }
  return *this;
}

TypedBuffer::TypedBuffer(TypedBuffer &&other) noexcept
    : mDataType(other.mDataType), mLocation(other.mLocation),
      mCpuData(std::move(other.mCpuData)), mDevicePtr(other.mDevicePtr),
      mDeviceBufferSizeBytes(other.mDeviceBufferSizeBytes),
      mDeviceId(other.mDeviceId), mElementCount(other.mElementCount) {
  other.reset();
}

TypedBuffer &TypedBuffer::operator=(TypedBuffer &&other) noexcept {
  if (this != &other) {
    mDataType = other.mDataType;
    mLocation = other.mLocation;
    mCpuData = std::move(other.mCpuData);
    mDevicePtr = other.mDevicePtr;
    mDeviceBufferSizeBytes = other.mDeviceBufferSizeBytes;
    mDeviceId = other.mDeviceId;
    mElementCount = other.mElementCount;
    other.reset();
  }
  return *this;
}

TypedBuffer TypedBuffer::createFromCpu(DataType type,
                                       const std::vector<uint8_t> &data) {
  return TypedBuffer(type, data);
}

TypedBuffer TypedBuffer::createFromCpu(DataType type,
                                       std::vector<uint8_t> &&data) {
  return TypedBuffer(type, std::move(data));
}

TypedBuffer TypedBuffer::createFromGpu(DataType type, void *devicePtr,
                                       size_t sizeBytes, int deviceId) {
  return TypedBuffer(type, devicePtr, sizeBytes, deviceId);
}

size_t TypedBuffer::getSizeBytes() const noexcept {
  return (mLocation == BufferLocation::CPU) ? mCpuData.size()
                                            : mDeviceBufferSizeBytes;
}

int TypedBuffer::getDeviceId() const noexcept {
  if (mLocation != BufferLocation::GPU_DEVICE) {
    return 0;
  }
  return mDeviceId;
}

const void *TypedBuffer::getRawHostPtr() const {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  return mCpuData.data();
}

void *TypedBuffer::getRawHostPtr() {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  return mCpuData.data();
}

void *TypedBuffer::getRawDevicePtr() const {
  if (mLocation != BufferLocation::GPU_DEVICE) {
    throw std::runtime_error(
        "Attempted to get device pointer from a non-GPU buffer.");
  }
  return mDevicePtr;
}

void TypedBuffer::setCpuData(DataType type, const std::vector<uint8_t> &data) {
  *this = createFromCpu(type, data);
}

void TypedBuffer::setCpuData(DataType type, std::vector<uint8_t> &&data) {
  *this = createFromCpu(type, std::move(data));
}

void TypedBuffer::setGpuDataReference(DataType type, void *ptr,
                                      size_t sizeBytes, int devId) {
  *this = createFromGpu(type, ptr, sizeBytes, devId);
}

void TypedBuffer::resize(size_t newElementCount) {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Cannot resize a buffer that references GPU memory.");
  }
  mElementCount = newElementCount;
  mCpuData.resize(mElementCount * getElementSize(mDataType));
}

void TypedBuffer::clear() { reset(); }

size_t TypedBuffer::getElementSize(DataType type) noexcept {
  switch (type) {
  case DataType::FLOAT32:
    return sizeof(float);
  case DataType::FLOAT16:
    return sizeof(uint16_t);
  case DataType::INT8:
    return sizeof(int8_t);
  default:
    return 0;
  }
}

TypedBuffer::TypedBuffer(DataType type, const std::vector<uint8_t> &cpuData)
    : mDataType(type), mLocation(BufferLocation::CPU), mCpuData(cpuData) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && !mCpuData.empty())
    throw std::runtime_error("Unsupported data type.");
  mElementCount = mCpuData.empty() ? 0 : mCpuData.size() / elemSize;
}

TypedBuffer::TypedBuffer(DataType type, std::vector<uint8_t> &&cpuData)
    : mDataType(type), mLocation(BufferLocation::CPU),
      mCpuData(std::move(cpuData)) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && !mCpuData.empty())
    throw std::runtime_error("Unsupported data type.");
  mElementCount = mCpuData.empty() ? 0 : mCpuData.size() / elemSize;
}

TypedBuffer::TypedBuffer(DataType type, void *devicePtr, size_t bufferSizeBytes,
                         int deviceId)
    : mDataType(type), mLocation(BufferLocation::GPU_DEVICE),
      mDevicePtr(devicePtr), mDeviceBufferSizeBytes(bufferSizeBytes),
      mDeviceId(deviceId) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && bufferSizeBytes > 0)
    throw std::runtime_error("Unsupported data type.");
  mElementCount = (mDevicePtr == nullptr || bufferSizeBytes == 0)
                      ? 0
                      : mDeviceBufferSizeBytes / elemSize;
}

void TypedBuffer::reset() {
  mDataType = DataType::FLOAT32;
  mLocation = BufferLocation::CPU;
  mCpuData.clear();
  mDevicePtr = nullptr;
  mDeviceBufferSizeBytes = 0;
  mDeviceId = 0;
  mElementCount = 0;
}

} // namespace ai_core