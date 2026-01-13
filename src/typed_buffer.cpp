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
  if (m_isExternalRef && m_manageExternalCpu && m_externalCpuPtr) {
    // Note: Assuming uint8_t array allocation for generic void*
    // In a real generic container, deleting void* is undefined behavior without
    // a custom deleter. Here we assume standard byte array usage.
    delete[] static_cast<uint8_t *>(m_externalCpuPtr);
  }

  // mAccelBuffer and mCpuData clean themselves up
  m_accelBuffer.reset();
  m_cpuData.clear();

  m_externalCpuPtr = nullptr;
  m_isExternalRef = false;
  m_manageExternalCpu = false;
  m_elementCount = 0;
  m_dataType = DataType::FLOAT32;
  m_location = BufferLocation::CPU;
  m_memoryType = BufferMemoryType::Pageable;
  m_deviceId = 0;
}

// Copy Constructor
TypedBuffer::TypedBuffer(const TypedBuffer &other)
    : m_dataType(other.m_dataType), m_location(other.m_location),
      m_memoryType(other.m_memoryType), m_deviceId(other.m_deviceId),
      m_elementCount(other.m_elementCount),
      // References are converted to deep copies by default in copy-ctor
      m_isExternalRef(false), m_manageExternalCpu(false),
      m_externalCpuPtr(nullptr) {

  // Handle CPU Pageable Data
  if (other.m_location == BufferLocation::CPU &&
      other.m_memoryType == BufferMemoryType::Pageable) {
    if (other.m_isExternalRef && other.m_externalCpuPtr) {
      // Deep copy external reference to internal vector
      size_t bytes = other.getSizeBytes();
      const uint8_t *src = static_cast<const uint8_t *>(other.getRawHostPtr());
      m_cpuData.assign(src, src + bytes);
    } else {
      m_cpuData = other.m_cpuData;
    }
  }

  // Handle Accelerator Data (GPU or Pinned)
  if (other.m_accelBuffer) {
    m_accelBuffer = AcceleratorBufferImpl::clone(*other.m_accelBuffer);
  }
}

// Copy Assignment
TypedBuffer &TypedBuffer::operator=(const TypedBuffer &other) {
  if (this != &other) {
    reset(); // Clean up current resources

    m_dataType = other.m_dataType;
    m_location = other.m_location;
    m_memoryType = other.m_memoryType;
    m_deviceId = other.m_deviceId;
    m_elementCount = other.m_elementCount;

    // Handle CPU Pageable
    if (other.m_location == BufferLocation::CPU &&
        other.m_memoryType == BufferMemoryType::Pageable) {
      if (other.m_isExternalRef && other.m_externalCpuPtr) {
        size_t bytes = other.getSizeBytes();
        const uint8_t *src =
            static_cast<const uint8_t *>(other.getRawHostPtr());
        m_cpuData.assign(src, src + bytes);
      } else {
        m_cpuData = other.m_cpuData;
      }
    }

    // Handle Accelerator Data
    if (other.m_accelBuffer) {
      m_accelBuffer = AcceleratorBufferImpl::clone(*other.m_accelBuffer);
    }
  }
  return *this;
}

// Move Constructor
TypedBuffer::TypedBuffer(TypedBuffer &&other) noexcept
    : m_dataType(other.m_dataType), m_location(other.m_location),
      m_memoryType(other.m_memoryType), m_elementCount(other.m_elementCount),
      m_cpuData(std::move(other.m_cpuData)),
      m_externalCpuPtr(other.m_externalCpuPtr),
      m_isExternalRef(other.m_isExternalRef),
      m_manageExternalCpu(other.m_manageExternalCpu),
      m_accelBuffer(std::move(other.m_accelBuffer)),
      m_deviceId(other.m_deviceId) {

  // Neutralize other
  other.m_externalCpuPtr = nullptr;
  other.m_isExternalRef = false;
  other.m_manageExternalCpu = false;
  other.m_elementCount = 0;
}

// Move Assignment
TypedBuffer &TypedBuffer::operator=(TypedBuffer &&other) noexcept {
  if (this != &other) {
    reset();

    m_dataType = other.m_dataType;
    m_location = other.m_location;
    m_memoryType = other.m_memoryType;
    m_elementCount = other.m_elementCount;
    m_deviceId = other.m_deviceId;

    m_cpuData = std::move(other.m_cpuData);
    m_accelBuffer = std::move(other.m_accelBuffer);

    m_externalCpuPtr = other.m_externalCpuPtr;
    m_isExternalRef = other.m_isExternalRef;
    m_manageExternalCpu = other.m_manageExternalCpu;

    // Neutralize other
    other.m_externalCpuPtr = nullptr;
    other.m_isExternalRef = false;
    other.m_manageExternalCpu = false;
    other.m_elementCount = 0;
  }
  return *this;
}

// ============================================================================
// Factory Implementation
// ============================================================================

TypedBuffer TypedBuffer::createFromCpu(DataType type,
                                       const std::vector<uint8_t> &data) {
  TypedBuffer buf;
  buf.m_dataType = type;
  buf.m_location = BufferLocation::CPU;
  buf.m_memoryType = BufferMemoryType::Pageable;
  buf.m_cpuData = data;
  size_t elem_size = getElementSize(type);
  buf.m_elementCount = (elem_size > 0) ? data.size() / elem_size : 0;
  return buf;
}

TypedBuffer TypedBuffer::createFromCpu(DataType type,
                                       std::vector<uint8_t> &&data) {
  TypedBuffer buf;
  buf.m_dataType = type;
  buf.m_location = BufferLocation::CPU;
  buf.m_memoryType = BufferMemoryType::Pageable;
  size_t size_bytes = data.size();
  buf.m_cpuData = std::move(data);
  size_t elem_size = getElementSize(type);
  buf.m_elementCount = (elem_size > 0) ? size_bytes / elem_size : 0;
  return buf;
}

TypedBuffer TypedBuffer::createFromCpuRef(DataType type, const void *host_ptr,
                                          size_t size_bytes,
                                          bool manage_memory) {
  TypedBuffer buf;
  buf.m_dataType = type;
  buf.m_location = BufferLocation::CPU;
  buf.m_memoryType = BufferMemoryType::Pageable;
  buf.m_externalCpuPtr = const_cast<void *>(host_ptr);
  buf.m_isExternalRef = true;
  buf.m_manageExternalCpu = manage_memory;
  size_t elem_size = getElementSize(type);
  buf.m_elementCount = (elem_size > 0) ? size_bytes / elem_size : 0;
  return buf;
}

TypedBuffer TypedBuffer::createFromGpu(DataType type, size_t size_bytes,
                                       int device_id) {
  TypedBuffer buf;
  buf.m_dataType = type;
  buf.m_location = BufferLocation::GpuDevice;
  // Device memory is typically pageable on the GPU, but managed by Accelerator
  buf.m_memoryType = BufferMemoryType::Pageable;
  buf.m_deviceId = device_id;
  size_t elem_size = getElementSize(type);
  buf.m_elementCount = (elem_size > 0) ? size_bytes / elem_size : 0;

  if (size_bytes > 0) {
    buf.m_accelBuffer = AcceleratorBufferImpl::create(
        size_bytes, AcceleratorMemoryType::Device);
  }
  return buf;
}

TypedBuffer TypedBuffer::createFromGpu(DataType type, void *device_ptr,
                                       size_t size_bytes, int device_id,
                                       bool manage_memory) {
  TypedBuffer buf;
  buf.m_dataType = type;
  buf.m_location = BufferLocation::GpuDevice;
  buf.m_memoryType = BufferMemoryType::Pageable;
  buf.m_deviceId = device_id;
  size_t elem_size = getElementSize(type);
  buf.m_elementCount = (elem_size > 0) ? size_bytes / elem_size : 0;

  if (size_bytes > 0 && device_ptr) {
    buf.m_accelBuffer = AcceleratorBufferImpl::createReference(
        device_ptr, size_bytes, AcceleratorMemoryType::Device, manage_memory);
  }
  return buf;
}

TypedBuffer TypedBuffer::createPinnedHost(DataType type, size_t size_bytes) {
  TypedBuffer buf;
  buf.m_dataType = type;
  buf.m_location = BufferLocation::CPU;
  buf.m_memoryType = BufferMemoryType::Pinned;
  size_t elem_size = getElementSize(type);
  buf.m_elementCount = (elem_size > 0) ? size_bytes / elem_size : 0;

  if (size_bytes > 0) {
    buf.m_accelBuffer = AcceleratorBufferImpl::create(
        size_bytes, AcceleratorMemoryType::HostPinned);
  }
  return buf;
}

// ============================================================================
// Property Queries
// ============================================================================

size_t TypedBuffer::getSizeBytes() const noexcept {
  // If backed by accelerator (Pinned or GPU), trust it
  if (m_accelBuffer) {
    return m_accelBuffer->getSizeBytes();
  }

  // Otherwise, standard CPU logic
  if (m_isExternalRef) {
    return m_elementCount * getElementSize(m_dataType);
  }
  return m_cpuData.size();
}

int TypedBuffer::getDeviceId() const noexcept {
  return (m_location == BufferLocation::GpuDevice) ? m_deviceId : 0;
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
  if (m_location != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to access Host pointer on Non-CPU buffer");
  }

  // 1. Check if it's Pinned Memory (held in AcceleratorBuffer)
  if (m_memoryType == BufferMemoryType::Pinned) {
    return m_accelBuffer ? m_accelBuffer->get() : nullptr;
  }

  // 2. Check External Ref
  if (m_isExternalRef) {
    return m_externalCpuPtr;
  }

  // 3. Standard Vector
  return m_cpuData.data();
}

void *TypedBuffer::getRawDevicePtr() const {
  if (m_location != BufferLocation::GpuDevice) {
    throw std::runtime_error(
        "Attempted to access Device pointer on Non-GPU buffer");
  }
  return m_accelBuffer ? m_accelBuffer->get() : nullptr;
}

// ============================================================================
// Modification
// ============================================================================

void TypedBuffer::setCpuData(DataType type, const std::vector<uint8_t> &data) {
  *this = createFromCpu(type, data);
}

void TypedBuffer::setGpuDataReference(DataType type, void *ptr,
                                      size_t size_bytes, int dev_id) {
  *this = createFromGpu(type, ptr, size_bytes, dev_id, false);
}

void TypedBuffer::clear() { reset(); }

void TypedBuffer::resize(size_t new_element_count) {
  if (new_element_count == m_elementCount)
    return;

  size_t new_size_bytes = new_element_count * getElementSize(m_dataType);

  // Case 1: Standard CPU Pageable
  if (m_location == BufferLocation::CPU &&
      m_memoryType == BufferMemoryType::Pageable) {
    // If it's an external reference, we must convert to owned to resize
    if (m_isExternalRef) {
      std::vector<uint8_t> new_data(new_size_bytes);
      if (m_externalCpuPtr && m_elementCount > 0) {
        size_t copy_size = std::min(getSizeBytes(), new_size_bytes);
        std::memcpy(new_data.data(), m_externalCpuPtr, copy_size);
      }
      // Reset ref and move to owned
      if (m_manageExternalCpu && m_externalCpuPtr) {
        delete[] static_cast<uint8_t *>(m_externalCpuPtr);
      }
      m_isExternalRef = false;
      m_externalCpuPtr = nullptr;
      m_manageExternalCpu = false;
      m_cpuData = std::move(new_data);
    } else {
      // Standard vector resize
      m_cpuData.resize(new_size_bytes);
    }
  }
  // Case 2: Accelerator Managed (Pinned CPU or Device GPU)
  else if (m_accelBuffer) {
    // Note: AcceleratorBuffer resizing typically implies reallocation.
    // Preserving data (copying old to new) is expensive and often unnecessary
    // for output buffers. If needed, clone-and-copy logic would go here.
    // Current Strategy: Destructive Resize (Reallocate)

    auto current_type =
        m_accelBuffer->getType(); // Needs getType() in interface
    m_accelBuffer = AcceleratorBufferImpl::create(new_size_bytes, current_type);
  }

  m_elementCount = new_element_count;
}

} // namespace ai_core