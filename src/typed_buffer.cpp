/**
 * @file typed_buffer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief TypedBuffer implementation with Pinned Memory support
 * @version 0.2
 * @date 2025-01-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/typed_buffer.hpp"
#include "ai_core/device_buffer_impl.hpp"
#include "ai_core/pinned_host_buffer_impl.hpp"
#include <cstring>

namespace ai_core {

TypedBuffer::TypedBuffer() = default;

TypedBuffer::~TypedBuffer() {
  if (mIsExternalRef && mManageExternalCpu && mExternalCpuPtr) {
    delete[] static_cast<uint8_t *>(mExternalCpuPtr);
    mExternalCpuPtr = nullptr;
  }
  // mPinnedImpl and mGpuImpl are automatically cleaned up by unique_ptr
}

TypedBuffer::TypedBuffer(const TypedBuffer &other)
    : mDataType(other.mDataType), mLocation(other.mLocation),
      mMemoryType(other.mMemoryType), mCpuData(other.mCpuData),
      mDeviceBufferSizeBytes(other.mDeviceBufferSizeBytes),
      mDeviceId(other.mDeviceId), mElementCount(other.mElementCount),
      mIsExternalRef(false), mManageExternalCpu(false),
      mExternalCpuPtr(nullptr) {

  // Handle external reference: copy data to owned storage
  if (other.mIsExternalRef && other.mExternalCpuPtr &&
      other.mElementCount > 0) {
    size_t sizeBytes = other.mElementCount * getElementSize(other.mDataType);
    mCpuData.assign(static_cast<const uint8_t *>(other.mExternalCpuPtr),
                    static_cast<const uint8_t *>(other.mExternalCpuPtr) +
                        sizeBytes);
    mMemoryType = BufferMemoryType::Pageable;
  }

  // Clone pinned memory
  if (other.mPinnedImpl) {
    mPinnedImpl = PinnedHostBufferImpl::clone(*other.mPinnedImpl);
  }

  // Clone GPU memory
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
    mMemoryType = other.mMemoryType;
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
      mMemoryType = BufferMemoryType::Pageable;
    }

    if (other.mPinnedImpl) {
      mPinnedImpl = PinnedHostBufferImpl::clone(*other.mPinnedImpl);
    } else {
      mPinnedImpl.reset();
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
      mMemoryType(other.mMemoryType), mCpuData(std::move(other.mCpuData)),
      mExternalCpuPtr(other.mExternalCpuPtr),
      mIsExternalRef(other.mIsExternalRef),
      mManageExternalCpu(other.mManageExternalCpu),
      mPinnedImpl(std::move(other.mPinnedImpl)),
      mGpuImpl(std::move(other.mGpuImpl)),
      mDeviceBufferSizeBytes(other.mDeviceBufferSizeBytes),
      mDeviceId(other.mDeviceId), mElementCount(other.mElementCount) {
  other.mExternalCpuPtr = nullptr;
  other.mIsExternalRef = false;
  other.mManageExternalCpu = false;
  other.mElementCount = 0;
  other.mMemoryType = BufferMemoryType::Pageable;
}

TypedBuffer &TypedBuffer::operator=(TypedBuffer &&other) noexcept {
  if (this != &other) {
    if (mIsExternalRef && mManageExternalCpu && mExternalCpuPtr) {
      delete[] static_cast<uint8_t *>(mExternalCpuPtr);
    }

    mDataType = other.mDataType;
    mLocation = other.mLocation;
    mMemoryType = other.mMemoryType;
    mCpuData = std::move(other.mCpuData);
    mExternalCpuPtr = other.mExternalCpuPtr;
    mIsExternalRef = other.mIsExternalRef;
    mManageExternalCpu = other.mManageExternalCpu;
    mPinnedImpl = std::move(other.mPinnedImpl);
    mGpuImpl = std::move(other.mGpuImpl);
    mDeviceBufferSizeBytes = other.mDeviceBufferSizeBytes;
    mDeviceId = other.mDeviceId;
    mElementCount = other.mElementCount;

    other.mExternalCpuPtr = nullptr;
    other.mIsExternalRef = false;
    other.mManageExternalCpu = false;
    other.mElementCount = 0;
    other.mMemoryType = BufferMemoryType::Pageable;
  }
  return *this;
}

// ============================================================================
// Private Constructors
// ============================================================================

TypedBuffer::TypedBuffer(DataType type, const std::vector<uint8_t> &cpuData)
    : mDataType(type), mLocation(BufferLocation::CPU),
      mMemoryType(BufferMemoryType::Pageable), mCpuData(cpuData),
      mExternalCpuPtr(nullptr), mIsExternalRef(false),
      mManageExternalCpu(false) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && !mCpuData.empty())
    throw std::runtime_error("Unsupported data type.");
  mElementCount = mCpuData.empty() ? 0 : mCpuData.size() / elemSize;
}

TypedBuffer::TypedBuffer(DataType type, std::vector<uint8_t> &&cpuData)
    : mDataType(type), mLocation(BufferLocation::CPU),
      mMemoryType(BufferMemoryType::Pageable), mCpuData(std::move(cpuData)),
      mExternalCpuPtr(nullptr), mIsExternalRef(false),
      mManageExternalCpu(false) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && !mCpuData.empty())
    throw std::runtime_error("Unsupported data type.");
  mElementCount = mCpuData.empty() ? 0 : mCpuData.size() / elemSize;
}

TypedBuffer::TypedBuffer(DataType type, size_t bufferSizeBytes, int deviceId)
    : mDataType(type), mLocation(BufferLocation::GPU_DEVICE),
      mMemoryType(BufferMemoryType::Pageable),
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
      mMemoryType(BufferMemoryType::Pageable),
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

TypedBuffer::TypedBuffer(DataType type, const void *hostPtr, size_t sizeBytes,
                         bool manageMemory)
    : mDataType(type), mLocation(BufferLocation::CPU),
      mMemoryType(BufferMemoryType::Pageable),
      mExternalCpuPtr(const_cast<void *>(hostPtr)), mIsExternalRef(true),
      mManageExternalCpu(manageMemory) {
  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && sizeBytes > 0)
    throw std::runtime_error("Unsupported data type.");
  mElementCount =
      (hostPtr == nullptr || sizeBytes == 0) ? 0 : sizeBytes / elemSize;
}

// Constructor for pinned memory
TypedBuffer::TypedBuffer(DataType type, size_t sizeBytes,
                         BufferMemoryType memType)
    : mDataType(type), mLocation(BufferLocation::CPU), mMemoryType(memType),
      mExternalCpuPtr(nullptr), mIsExternalRef(false),
      mManageExternalCpu(false) {
  if (memType == BufferMemoryType::Pinned) {
    mPinnedImpl = PinnedHostBufferImpl::create(sizeBytes);
    if (!mPinnedImpl) {
      throw std::runtime_error("Failed to create pinned memory buffer");
    }
  } else {
    throw std::runtime_error(
        "Invalid memory type for this constructor. Use Pinned.");
  }

  const size_t elemSize = getElementSize(mDataType);
  if (elemSize == 0 && sizeBytes > 0)
    throw std::runtime_error("Unsupported data type.");
  mElementCount = sizeBytes == 0 ? 0 : sizeBytes / elemSize;
}

// ============================================================================
// Factory Methods
// ============================================================================

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

TypedBuffer TypedBuffer::createPinnedHost(DataType type, size_t sizeBytes) {
  return TypedBuffer(type, sizeBytes, BufferMemoryType::Pinned);
}

// ============================================================================
// Property Queries
// ============================================================================

size_t TypedBuffer::getSizeBytes() const noexcept {
  if (mLocation == BufferLocation::CPU) {
    if (mPinnedImpl) {
      return mPinnedImpl->getSizeBytes();
    }
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

// ============================================================================
// Data Access
// ============================================================================

const void *TypedBuffer::getRawHostPtr() const {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }

  if (mPinnedImpl) {
    return mPinnedImpl->get();
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

  if (mPinnedImpl) {
    return mPinnedImpl->get();
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

// ============================================================================
// Data Modification
// ============================================================================

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
    // Cannot resize pinned memory in-place, need to reallocate
    if (mPinnedImpl) {
      size_t newSizeBytes = newElementCount * getElementSize(mDataType);
      size_t oldSizeBytes = mPinnedImpl->getSizeBytes();

      if (newSizeBytes != oldSizeBytes) {
        auto newPinned = PinnedHostBufferImpl::create(newSizeBytes);
        if (newPinned && mPinnedImpl->get() && newPinned->get()) {
          size_t copySize = std::min(oldSizeBytes, newSizeBytes);
          std::memcpy(newPinned->get(), mPinnedImpl->get(), copySize);
        }
        mPinnedImpl = std::move(newPinned);
      }
      mElementCount = newElementCount;
      return;
    }

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
      mMemoryType = BufferMemoryType::Pageable;
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
  mMemoryType = BufferMemoryType::Pageable;
  mCpuData.clear();
  mExternalCpuPtr = nullptr;
  mIsExternalRef = false;
  mManageExternalCpu = false;
  mPinnedImpl.reset();
  mGpuImpl.reset();
  mDeviceBufferSizeBytes = 0;
  mDeviceId = 0;
  mElementCount = 0;
}

} // namespace ai_core