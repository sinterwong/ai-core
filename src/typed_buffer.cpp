/**
 * @file typed_buffer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Implementation of TypedBuffer with unified backend storage
 * @version 0.3
 * @date 2026-01-06
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "ai_core/typed_buffer.hpp"
#include <algorithm>
#include <cstring>

namespace ai_core {

// ============================================================================
// Lifecycle Management
// ============================================================================

TypedBuffer::TypedBuffer() = default;

TypedBuffer::~TypedBuffer() { reset(); }

void TypedBuffer::reset() {
  // Clean up manually managed external pointer
  if (mIsExternalRef && mManageExternalCpu && mExternalCpuPtr) {
    // Note: Assuming uint8_t array allocation for generic void*
    // In a real generic container, deleting void* is undefined behavior without
    // a custom deleter. Here we assume standard byte array usage.
    delete[] static_cast<uint8_t *>(mExternalCpuPtr);
  }

  // mAccelBuffer and mCpuData clean themselves up
  mAccelBuffer.reset();
  mCpuData.clear();

  mExternalCpuPtr = nullptr;
  mIsExternalRef = false;
  mManageExternalCpu = false;
  mElementCount = 0;
  mDataType = DataType::FLOAT32;
  mLocation = BufferLocation::CPU;
  mMemoryType = BufferMemoryType::Pageable;
  mDeviceId = 0;
}

// Copy Constructor
TypedBuffer::TypedBuffer(const TypedBuffer &other)
    : mDataType(other.mDataType), mLocation(other.mLocation),
      mMemoryType(other.mMemoryType), mDeviceId(other.mDeviceId),
      mElementCount(other.mElementCount),
      // References are converted to deep copies by default in copy-ctor
      mIsExternalRef(false), mManageExternalCpu(false),
      mExternalCpuPtr(nullptr) {

  // Handle CPU Pageable Data
  if (other.mLocation == BufferLocation::CPU &&
      other.mMemoryType == BufferMemoryType::Pageable) {
    if (other.mIsExternalRef && other.mExternalCpuPtr) {
      // Deep copy external reference to internal vector
      size_t bytes = other.getSizeBytes();
      const uint8_t *src = static_cast<const uint8_t *>(other.getRawHostPtr());
      mCpuData.assign(src, src + bytes);
    } else {
      mCpuData = other.mCpuData;
    }
  }

  // Handle Accelerator Data (GPU or Pinned)
  if (other.mAccelBuffer) {
    mAccelBuffer = AcceleratorBufferImpl::clone(*other.mAccelBuffer);
  }
}

// Copy Assignment
TypedBuffer &TypedBuffer::operator=(const TypedBuffer &other) {
  if (this != &other) {
    reset(); // Clean up current resources

    mDataType = other.mDataType;
    mLocation = other.mLocation;
    mMemoryType = other.mMemoryType;
    mDeviceId = other.mDeviceId;
    mElementCount = other.mElementCount;

    // Handle CPU Pageable
    if (other.mLocation == BufferLocation::CPU &&
        other.mMemoryType == BufferMemoryType::Pageable) {
      if (other.mIsExternalRef && other.mExternalCpuPtr) {
        size_t bytes = other.getSizeBytes();
        const uint8_t *src =
            static_cast<const uint8_t *>(other.getRawHostPtr());
        mCpuData.assign(src, src + bytes);
      } else {
        mCpuData = other.mCpuData;
      }
    }

    // Handle Accelerator Data
    if (other.mAccelBuffer) {
      mAccelBuffer = AcceleratorBufferImpl::clone(*other.mAccelBuffer);
    }
  }
  return *this;
}

// Move Constructor
TypedBuffer::TypedBuffer(TypedBuffer &&other) noexcept
    : mDataType(other.mDataType), mLocation(other.mLocation),
      mMemoryType(other.mMemoryType), mElementCount(other.mElementCount),
      mCpuData(std::move(other.mCpuData)),
      mExternalCpuPtr(other.mExternalCpuPtr),
      mIsExternalRef(other.mIsExternalRef),
      mManageExternalCpu(other.mManageExternalCpu),
      mAccelBuffer(std::move(other.mAccelBuffer)), mDeviceId(other.mDeviceId) {

  // Neutralize other
  other.mExternalCpuPtr = nullptr;
  other.mIsExternalRef = false;
  other.mManageExternalCpu = false;
  other.mElementCount = 0;
}

// Move Assignment
TypedBuffer &TypedBuffer::operator=(TypedBuffer &&other) noexcept {
  if (this != &other) {
    reset();

    mDataType = other.mDataType;
    mLocation = other.mLocation;
    mMemoryType = other.mMemoryType;
    mElementCount = other.mElementCount;
    mDeviceId = other.mDeviceId;

    mCpuData = std::move(other.mCpuData);
    mAccelBuffer = std::move(other.mAccelBuffer);

    mExternalCpuPtr = other.mExternalCpuPtr;
    mIsExternalRef = other.mIsExternalRef;
    mManageExternalCpu = other.mManageExternalCpu;

    // Neutralize other
    other.mExternalCpuPtr = nullptr;
    other.mIsExternalRef = false;
    other.mManageExternalCpu = false;
    other.mElementCount = 0;
  }
  return *this;
}

// ============================================================================
// Factory Implementation
// ============================================================================

TypedBuffer TypedBuffer::createFromCpu(DataType type,
                                       const std::vector<uint8_t> &data) {
  TypedBuffer buf;
  buf.mDataType = type;
  buf.mLocation = BufferLocation::CPU;
  buf.mMemoryType = BufferMemoryType::Pageable;
  buf.mCpuData = data;
  size_t elemSize = getElementSize(type);
  buf.mElementCount = (elemSize > 0) ? data.size() / elemSize : 0;
  return buf;
}

TypedBuffer TypedBuffer::createFromCpu(DataType type,
                                       std::vector<uint8_t> &&data) {
  TypedBuffer buf;
  buf.mDataType = type;
  buf.mLocation = BufferLocation::CPU;
  buf.mMemoryType = BufferMemoryType::Pageable;
  size_t sizeBytes = data.size();
  buf.mCpuData = std::move(data);
  size_t elemSize = getElementSize(type);
  buf.mElementCount = (elemSize > 0) ? sizeBytes / elemSize : 0;
  return buf;
}

TypedBuffer TypedBuffer::createFromCpuRef(DataType type, const void *hostPtr,
                                          size_t sizeBytes, bool manageMemory) {
  TypedBuffer buf;
  buf.mDataType = type;
  buf.mLocation = BufferLocation::CPU;
  buf.mMemoryType = BufferMemoryType::Pageable;
  buf.mExternalCpuPtr = const_cast<void *>(hostPtr);
  buf.mIsExternalRef = true;
  buf.mManageExternalCpu = manageMemory;
  size_t elemSize = getElementSize(type);
  buf.mElementCount = (elemSize > 0) ? sizeBytes / elemSize : 0;
  return buf;
}

TypedBuffer TypedBuffer::createFromGpu(DataType type, size_t sizeBytes,
                                       int deviceId) {
  TypedBuffer buf;
  buf.mDataType = type;
  buf.mLocation = BufferLocation::GPU_DEVICE;
  // Device memory is typically pageable on the GPU, but managed by Accelerator
  buf.mMemoryType = BufferMemoryType::Pageable;
  buf.mDeviceId = deviceId;
  size_t elemSize = getElementSize(type);
  buf.mElementCount = (elemSize > 0) ? sizeBytes / elemSize : 0;

  if (sizeBytes > 0) {
    buf.mAccelBuffer =
        AcceleratorBufferImpl::create(sizeBytes, AcceleratorMemoryType::Device);
  }
  return buf;
}

TypedBuffer TypedBuffer::createFromGpu(DataType type, void *devicePtr,
                                       size_t sizeBytes, int deviceId,
                                       bool manageMemory) {
  TypedBuffer buf;
  buf.mDataType = type;
  buf.mLocation = BufferLocation::GPU_DEVICE;
  buf.mMemoryType = BufferMemoryType::Pageable;
  buf.mDeviceId = deviceId;
  size_t elemSize = getElementSize(type);
  buf.mElementCount = (elemSize > 0) ? sizeBytes / elemSize : 0;

  if (sizeBytes > 0 && devicePtr) {
    buf.mAccelBuffer = AcceleratorBufferImpl::createReference(
        devicePtr, sizeBytes, AcceleratorMemoryType::Device, manageMemory);
  }
  return buf;
}

TypedBuffer TypedBuffer::createPinnedHost(DataType type, size_t sizeBytes) {
  TypedBuffer buf;
  buf.mDataType = type;
  buf.mLocation = BufferLocation::CPU;
  buf.mMemoryType = BufferMemoryType::Pinned;
  size_t elemSize = getElementSize(type);
  buf.mElementCount = (elemSize > 0) ? sizeBytes / elemSize : 0;

  if (sizeBytes > 0) {
    buf.mAccelBuffer = AcceleratorBufferImpl::create(
        sizeBytes, AcceleratorMemoryType::HostPinned);
  }
  return buf;
}

// ============================================================================
// Property Queries
// ============================================================================

size_t TypedBuffer::getSizeBytes() const noexcept {
  // If backed by accelerator (Pinned or GPU), trust it
  if (mAccelBuffer) {
    return mAccelBuffer->getSizeBytes();
  }

  // Otherwise, standard CPU logic
  if (mIsExternalRef) {
    return mElementCount * getElementSize(mDataType);
  }
  return mCpuData.size();
}

int TypedBuffer::getDeviceId() const noexcept {
  return (mLocation == BufferLocation::GPU_DEVICE) ? mDeviceId : 0;
}

size_t TypedBuffer::getElementSize(DataType type) noexcept {
  switch (type) {
  case DataType::FLOAT32:
    return sizeof(float);
  case DataType::FLOAT16:
    return 2;
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

// ============================================================================
// Data Access
// ============================================================================

const void *TypedBuffer::getRawHostPtr() const {
  return const_cast<TypedBuffer *>(this)->getRawHostPtr();
}

void *TypedBuffer::getRawHostPtr() {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to access Host pointer on Non-CPU buffer");
  }

  // 1. Check if it's Pinned Memory (held in AcceleratorBuffer)
  if (mMemoryType == BufferMemoryType::Pinned) {
    return mAccelBuffer ? mAccelBuffer->get() : nullptr;
  }

  // 2. Check External Ref
  if (mIsExternalRef) {
    return mExternalCpuPtr;
  }

  // 3. Standard Vector
  return mCpuData.data();
}

void *TypedBuffer::getRawDevicePtr() const {
  if (mLocation != BufferLocation::GPU_DEVICE) {
    throw std::runtime_error(
        "Attempted to access Device pointer on Non-GPU buffer");
  }
  return mAccelBuffer ? mAccelBuffer->get() : nullptr;
}

// ============================================================================
// Modification
// ============================================================================

void TypedBuffer::setCpuData(DataType type, const std::vector<uint8_t> &data) {
  *this = createFromCpu(type, data);
}

void TypedBuffer::setGpuDataReference(DataType type, void *ptr,
                                      size_t sizeBytes, int devId) {
  *this = createFromGpu(type, ptr, sizeBytes, devId, false);
}

void TypedBuffer::clear() { reset(); }

void TypedBuffer::resize(size_t newElementCount) {
  if (newElementCount == mElementCount)
    return;

  size_t newSizeBytes = newElementCount * getElementSize(mDataType);

  // Case 1: Standard CPU Pageable
  if (mLocation == BufferLocation::CPU &&
      mMemoryType == BufferMemoryType::Pageable) {
    // If it's an external reference, we must convert to owned to resize
    if (mIsExternalRef) {
      std::vector<uint8_t> newData(newSizeBytes);
      if (mExternalCpuPtr && mElementCount > 0) {
        size_t copySize = std::min(getSizeBytes(), newSizeBytes);
        std::memcpy(newData.data(), mExternalCpuPtr, copySize);
      }
      // Reset ref and move to owned
      if (mManageExternalCpu && mExternalCpuPtr) {
        delete[] static_cast<uint8_t *>(mExternalCpuPtr);
      }
      mIsExternalRef = false;
      mExternalCpuPtr = nullptr;
      mManageExternalCpu = false;
      mCpuData = std::move(newData);
    } else {
      // Standard vector resize
      mCpuData.resize(newSizeBytes);
    }
  }
  // Case 2: Accelerator Managed (Pinned CPU or Device GPU)
  else if (mAccelBuffer) {
    // Note: AcceleratorBuffer resizing typically implies reallocation.
    // Preserving data (copying old to new) is expensive and often unnecessary
    // for output buffers. If needed, clone-and-copy logic would go here.
    // Current Strategy: Destructive Resize (Reallocate)

    auto currentType = mAccelBuffer->getType(); // Needs getType() in interface
    mAccelBuffer = AcceleratorBufferImpl::create(newSizeBytes, currentType);
  }

  mElementCount = newElementCount;
}

} // namespace ai_core