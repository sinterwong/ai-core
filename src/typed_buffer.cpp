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
#include "ai_core/device_buffer_impl.hpp"

namespace ai_core {

TypedBuffer::TypedBuffer() = default;

TypedBuffer::~TypedBuffer() {
  if (mIsExternalRef && mManageExternalCpu && mExternalCpuPtr) {
    delete[] static_cast<uint8_t *>(mExternalCpuPtr);
    mExternalCpuPtr = nullptr;
  }
}

TypedBuffer::TypedBuffer(const TypedBuffer &other)
    : mDataType(other.mDataType), mLocation(other.mLocation),
      mCpuData(other.mCpuData),
      mDeviceBufferSizeBytes(other.mDeviceBufferSizeBytes),
      mDeviceId(other.mDeviceId), mElementCount(other.mElementCount),
      mIsExternalRef(false), mManageExternalCpu(false),
      mExternalCpuPtr(nullptr) {

  if (other.mIsExternalRef && other.mExternalCpuPtr &&
      other.mElementCount > 0) {
    size_t sizeBytes = other.mElementCount * getElementSize(other.mDataType);
    mCpuData.assign(static_cast<const uint8_t *>(other.mExternalCpuPtr),
                    static_cast<const uint8_t *>(other.mExternalCpuPtr) +
                        sizeBytes);
  }

  if (other.mGpuImpl) {
    mGpuImpl = DeviceBufferImpl::clone(*other.mGpuImpl);
  }
}

TypedBuffer &TypedBuffer::operator=(const TypedBuffer &other) {
  if (this != &other) {
    if (mIsExternalRef && mManageExternalCpu && mExternalCpuPtr) {
      delete[] static_cast<uint8_t *>(mExternalCpuPtr);
    }

    mDataType = other.mDataType;
    mLocation = other.mLocation;
    mCpuData = other.mCpuData;
    mDeviceBufferSizeBytes = other.mDeviceBufferSizeBytes;
    mDeviceId = other.mDeviceId;
    mElementCount = other.mElementCount;
    mIsExternalRef = false;
    mManageExternalCpu = false;
    mExternalCpuPtr = nullptr;

    if (other.mIsExternalRef && other.mExternalCpuPtr &&
        other.mElementCount > 0) {
      size_t sizeBytes = other.mElementCount * getElementSize(other.mDataType);
      mCpuData.assign(static_cast<const uint8_t *>(other.mExternalCpuPtr),
                      static_cast<const uint8_t *>(other.mExternalCpuPtr) +
                          sizeBytes);
    }

    if (other.mGpuImpl) {
      mGpuImpl = DeviceBufferImpl::clone(*other.mGpuImpl);
    } else {
      mGpuImpl.reset();
    }
  }
  return *this;
}

TypedBuffer::TypedBuffer(TypedBuffer &&other) noexcept
    : mDataType(other.mDataType), mLocation(other.mLocation),
      mCpuData(std::move(other.mCpuData)),
      mExternalCpuPtr(other.mExternalCpuPtr),
      mIsExternalRef(other.mIsExternalRef),
      mManageExternalCpu(other.mManageExternalCpu),
      mGpuImpl(std::move(other.mGpuImpl)),
      mDeviceBufferSizeBytes(other.mDeviceBufferSizeBytes),
      mDeviceId(other.mDeviceId), mElementCount(other.mElementCount) {
  other.mExternalCpuPtr = nullptr;
  other.mIsExternalRef = false;
  other.mManageExternalCpu = false;
  other.mElementCount = 0;
}

TypedBuffer &TypedBuffer::operator=(TypedBuffer &&other) noexcept {
  if (this != &other) {
    if (mIsExternalRef && mManageExternalCpu && mExternalCpuPtr) {
      delete[] static_cast<uint8_t *>(mExternalCpuPtr);
    }

    mDataType = other.mDataType;
    mLocation = other.mLocation;
    mCpuData = std::move(other.mCpuData);
    mExternalCpuPtr = other.mExternalCpuPtr;
    mIsExternalRef = other.mIsExternalRef;
    mManageExternalCpu = other.mManageExternalCpu;
    mGpuImpl = std::move(other.mGpuImpl);
    mDeviceBufferSizeBytes = other.mDeviceBufferSizeBytes;
    mDeviceId = other.mDeviceId;
    mElementCount = other.mElementCount;

    // 清除源对象的外部指针所有权
    other.mExternalCpuPtr = nullptr;
    other.mIsExternalRef = false;
    other.mManageExternalCpu = false;
    other.mElementCount = 0;
  }
  return *this;
}

TypedBuffer::TypedBuffer(DataType type, const std::vector<uint8_t> &cpuData)
    : mDataType(type), mLocation(BufferLocation::CPU), mCpuData(cpuData),
      mExternalCpuPtr(nullptr), mIsExternalRef(false),
      mManageExternalCpu(false) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && !mCpuData.empty())
    throw std::runtime_error("Unsupported data type.");
  mElementCount = mCpuData.empty() ? 0 : mCpuData.size() / elemSize;
}

TypedBuffer::TypedBuffer(DataType type, std::vector<uint8_t> &&cpuData)
    : mDataType(type), mLocation(BufferLocation::CPU),
      mCpuData(std::move(cpuData)), mExternalCpuPtr(nullptr),
      mIsExternalRef(false), mManageExternalCpu(false) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && !mCpuData.empty())
    throw std::runtime_error("Unsupported data type.");
  mElementCount = mCpuData.empty() ? 0 : mCpuData.size() / elemSize;
}

TypedBuffer::TypedBuffer(DataType type, size_t bufferSizeBytes, int deviceId)
    : mDataType(type), mLocation(BufferLocation::GPU_DEVICE),
      mDeviceBufferSizeBytes(bufferSizeBytes), mDeviceId(deviceId),
      mExternalCpuPtr(nullptr), mIsExternalRef(false),
      mManageExternalCpu(false) {
  mGpuImpl = DeviceBufferImpl::create(bufferSizeBytes);
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
      mDeviceBufferSizeBytes(bufferSizeBytes), mDeviceId(deviceId),
      mExternalCpuPtr(nullptr), mIsExternalRef(false),
      mManageExternalCpu(false) {
  mGpuImpl = DeviceBufferImpl::create(devicePtr, bufferSizeBytes, manageMemory);
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && bufferSizeBytes > 0)
    throw std::runtime_error("Unsupported data type.");
  mElementCount = (mGpuImpl == nullptr || bufferSizeBytes == 0)
                      ? 0
                      : bufferSizeBytes / elemSize;
}

// 新增：外部 CPU 内存构造函数
TypedBuffer::TypedBuffer(DataType type, const void *hostPtr, size_t sizeBytes,
                         bool manageMemory)
    : mDataType(type), mLocation(BufferLocation::CPU),
      mExternalCpuPtr(const_cast<void *>(hostPtr)), mIsExternalRef(true),
      mManageExternalCpu(manageMemory) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && sizeBytes > 0)
    throw std::runtime_error("Unsupported data type.");
  mElementCount =
      (hostPtr == nullptr || sizeBytes == 0) ? 0 : sizeBytes / elemSize;
}

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

TypedBuffer TypedBuffer::createFromCpuRef(DataType type, const void *hostPtr,
                                          size_t sizeBytes, bool manageMemory) {
  return TypedBuffer(type, hostPtr, sizeBytes, manageMemory);
}

size_t TypedBuffer::getSizeBytes() const noexcept {
  if (mLocation == BufferLocation::CPU) {
    // 外部引用时使用 elementCount 计算
    if (mIsExternalRef) {
      return mElementCount * getElementSize(mDataType);
    }
    return mCpuData.size();
  }
  return mDeviceBufferSizeBytes;
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
  if (mIsExternalRef && mExternalCpuPtr) {
    return mExternalCpuPtr;
  }
  return mCpuData.data();
}

void *TypedBuffer::getRawHostPtr() {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  if (mIsExternalRef && mExternalCpuPtr) {
    return mExternalCpuPtr;
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
    if (mIsExternalRef) {
      if (mExternalCpuPtr && mElementCount > 0) {
        size_t oldSize = mElementCount * getElementSize(mDataType);
        mCpuData.assign(static_cast<uint8_t *>(mExternalCpuPtr),
                        static_cast<uint8_t *>(mExternalCpuPtr) + oldSize);
      }
      if (mManageExternalCpu && mExternalCpuPtr) {
        delete[] static_cast<uint8_t *>(mExternalCpuPtr);
      }
      mExternalCpuPtr = nullptr;
      mIsExternalRef = false;
      mManageExternalCpu = false;
    }
    mElementCount = newElementCount;
    mCpuData.resize(mElementCount * getElementSize(mDataType));
  } else {
    size_t newSize = newElementCount * getElementSize(mDataType);
    if (newSize > mDeviceBufferSizeBytes) {
      mGpuImpl = DeviceBufferImpl::create(newSize);
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
  case DataType::INT32:
    return sizeof(int32_t);
  case DataType::INT64:
    return sizeof(int64_t);
  case DataType::INT8:
    return sizeof(int8_t);
  default:
    return 0;
  }
}

void TypedBuffer::reset() {
  if (mIsExternalRef && mManageExternalCpu && mExternalCpuPtr) {
    delete[] static_cast<uint8_t *>(mExternalCpuPtr);
  }

  mDataType = DataType::FLOAT32;
  mLocation = BufferLocation::CPU;
  mCpuData.clear();
  mExternalCpuPtr = nullptr;
  mIsExternalRef = false;
  mManageExternalCpu = false;
  mGpuImpl.reset();
  mDeviceBufferSizeBytes = 0;
  mDeviceId = 0;
  mElementCount = 0;
}

} // namespace ai_core
