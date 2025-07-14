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
#include "trt/gpu_buffer_impl.hpp"
#include <logger.hpp>

namespace ai_core {

TypedBuffer::TypedBuffer() = default;
TypedBuffer::~TypedBuffer() = default;

TypedBuffer::TypedBuffer(const TypedBuffer &other)
    : mDataType(other.mDataType), mLocation(other.mLocation),
      mCpuData(other.mCpuData),
      mDeviceBufferSizeBytes(other.mDeviceBufferSizeBytes),
      mDeviceId(other.mDeviceId), mElementCount(other.mElementCount) {
  if (other.mGpuImpl) {
    mGpuImpl = std::make_unique<GpuBufferImpl>(*other.mGpuImpl);
  }
}

TypedBuffer &TypedBuffer::operator=(const TypedBuffer &other) {
  if (this != &other) {
    mDataType = other.mDataType;
    mLocation = other.mLocation;
    mCpuData = other.mCpuData;
    mDeviceBufferSizeBytes = other.mDeviceBufferSizeBytes;
    mDeviceId = other.mDeviceId;
    mElementCount = other.mElementCount;
    if (other.mGpuImpl) {
      mGpuImpl = std::make_unique<GpuBufferImpl>(*other.mGpuImpl);
    } else {
      mGpuImpl.reset();
    }
  }
  return *this;
}

TypedBuffer::TypedBuffer(TypedBuffer &&other) noexcept = default;
TypedBuffer &TypedBuffer::operator=(TypedBuffer &&other) noexcept = default;

TypedBuffer TypedBuffer::createFromCpu(DataType type,
                                       const std::vector<uint8_t> &data) {
  return TypedBuffer(type, data);
}

TypedBuffer TypedBuffer::createFromCpu(DataType type,
                                       std::vector<uint8_t> &&data) {
  return TypedBuffer(type, std::move(data));
}

TypedBuffer TypedBuffer::createFromGpu(DataType type, size_t sizeBytes,
                                       int deviceId) {
  return TypedBuffer(type, sizeBytes, deviceId);
}

TypedBuffer TypedBuffer::createFromGpu(DataType type, void *devicePtr,
                                       size_t sizeBytes, int deviceId,
                                       bool manageMemory) {
  return TypedBuffer(type, devicePtr, sizeBytes, deviceId, manageMemory);
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
  return mGpuImpl ? mGpuImpl->get() : nullptr;
}

void TypedBuffer::setCpuData(DataType type, const std::vector<uint8_t> &data) {
  *this = createFromCpu(type, data);
}

void TypedBuffer::setCpuData(DataType type, std::vector<uint8_t> &&data) {
  *this = createFromCpu(type, std::move(data));
}

void TypedBuffer::setGpuDataReference(DataType type, void *ptr,
                                      size_t sizeBytes, int devId) {
  *this = createFromGpu(type, ptr, sizeBytes, devId, false);
}

void TypedBuffer::resize(size_t newElementCount) {
  if (mLocation == BufferLocation::CPU) {
    mElementCount = newElementCount;
    mCpuData.resize(mElementCount * getElementSize(mDataType));
  } else {
    size_t newSize = newElementCount * getElementSize(mDataType);
    if (newSize > mDeviceBufferSizeBytes) {
      mGpuImpl = std::make_unique<GpuBufferImpl>(newSize);
      mDeviceBufferSizeBytes = newSize;
    }
    mElementCount = newElementCount;
  }
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

TypedBuffer::TypedBuffer(DataType type, size_t bufferSizeBytes, int deviceId)
    : mDataType(type), mLocation(BufferLocation::GPU_DEVICE),
      mDeviceBufferSizeBytes(bufferSizeBytes), mDeviceId(deviceId) {
  mGpuImpl = std::make_unique<GpuBufferImpl>(bufferSizeBytes);
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && bufferSizeBytes > 0)
    throw std::runtime_error("Unsupported data type.");
  mElementCount = (mGpuImpl == nullptr || bufferSizeBytes == 0)
                      ? 0
                      : bufferSizeBytes / elemSize;
}

TypedBuffer::TypedBuffer(DataType type, void *devicePtr, size_t bufferSizeBytes,
                         int deviceId, bool manageMemory)
    : mDataType(type), mLocation(BufferLocation::GPU_DEVICE),
      mDeviceBufferSizeBytes(bufferSizeBytes), mDeviceId(deviceId) {
  mGpuImpl =
      std::make_unique<GpuBufferImpl>(devicePtr, bufferSizeBytes, manageMemory);
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && bufferSizeBytes > 0)
    throw std::runtime_error("Unsupported data type.");
  mElementCount = (mGpuImpl == nullptr || bufferSizeBytes == 0)
                      ? 0
                      : bufferSizeBytes / elemSize;
}

void TypedBuffer::reset() {
  mDataType = DataType::FLOAT32;
  mLocation = BufferLocation::CPU;
  mCpuData.clear();
  mGpuImpl.reset();
  mDeviceBufferSizeBytes = 0;
  mDeviceId = 0;
  mElementCount = 0;
}

} // namespace ai_core
