#include "ai_core/typed_buffer.hpp"
#include <algorithm>
#include <cstring>

namespace ai_core {

TypedBuffer::TypedBuffer() = default;

TypedBuffer::~TypedBuffer() { reset(); }

void TypedBuffer::reset() {
  // External pointers are observed but never freed.
  m_accelBuffer.reset();
  m_cpuData.clear();

  m_externalCpuPtr = nullptr;
  m_isExternalRef = false;
  m_elementCount = 0;
  m_dataType = DataType::FLOAT32;
  m_location = BufferLocation::CPU;
  m_memoryType = BufferMemoryType::Pageable;
  m_deviceId = 0;
}

TypedBuffer::TypedBuffer(const TypedBuffer &other)
    : m_dataType(other.m_dataType), m_location(other.m_location),
      m_memoryType(other.m_memoryType), m_deviceId(other.m_deviceId),
      m_elementCount(other.m_elementCount),
      // Copying a view intentionally detaches it into owned storage.
      m_isExternalRef(false), m_externalCpuPtr(nullptr) {

  if (other.m_location == BufferLocation::CPU &&
      other.m_memoryType == BufferMemoryType::Pageable) {
    if (other.m_isExternalRef && other.m_externalCpuPtr) {
      size_t bytes = other.getSizeBytes();
      const uint8_t *src = static_cast<const uint8_t *>(other.getRawHostPtr());
      m_cpuData.assign(src, src + bytes);
    } else {
      m_cpuData = other.m_cpuData;
    }
  }

  if (other.m_accelBuffer) {
    m_accelBuffer = other.m_accelBuffer->clone();
  }
}

TypedBuffer &TypedBuffer::operator=(const TypedBuffer &other) {
  if (this != &other) {
    reset();

    m_dataType = other.m_dataType;
    m_location = other.m_location;
    m_memoryType = other.m_memoryType;
    m_deviceId = other.m_deviceId;
    m_elementCount = other.m_elementCount;

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

    if (other.m_accelBuffer) {
      m_accelBuffer = other.m_accelBuffer->clone();
    }
  }
  return *this;
}

TypedBuffer::TypedBuffer(TypedBuffer &&other) noexcept
    : m_dataType(other.m_dataType), m_location(other.m_location),
      m_memoryType(other.m_memoryType), m_elementCount(other.m_elementCount),
      m_cpuData(std::move(other.m_cpuData)),
      m_externalCpuPtr(other.m_externalCpuPtr),
      m_isExternalRef(other.m_isExternalRef),
      m_accelBuffer(std::move(other.m_accelBuffer)),
      m_deviceId(other.m_deviceId) {

  other.m_externalCpuPtr = nullptr;
  other.m_isExternalRef = false;
  other.m_elementCount = 0;
}

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

    other.m_externalCpuPtr = nullptr;
    other.m_isExternalRef = false;
    other.m_elementCount = 0;
  }
  return *this;
}

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

TypedBuffer TypedBuffer::wrapCpu(DataType type, const void *host_ptr,
                                 size_t size_bytes) {
  TypedBuffer buf;
  buf.m_dataType = type;
  buf.m_location = BufferLocation::CPU;
  buf.m_memoryType = BufferMemoryType::Pageable;
  buf.m_externalCpuPtr = const_cast<void *>(host_ptr);
  buf.m_isExternalRef = true;
  size_t elem_size = getElementSize(type);
  buf.m_elementCount = (elem_size > 0) ? size_bytes / elem_size : 0;
  return buf;
}

TypedBuffer TypedBuffer::fromStorage(DataType type,
                                     std::unique_ptr<IBufferStorage> storage) {
  if (!storage) {
    throw std::invalid_argument("TypedBuffer storage must not be null");
  }
  TypedBuffer buffer;
  buffer.m_dataType = type;
  const auto descriptor = storage->descriptor();
  buffer.m_location = descriptor.kind == MemoryKind::Device
                          ? BufferLocation::GpuDevice
                          : BufferLocation::CPU;
  buffer.m_memoryType =
      descriptor.kind == MemoryKind::HostPinned ? BufferMemoryType::Pinned
      : descriptor.kind == MemoryKind::Unified  ? BufferMemoryType::Managed
                                                : BufferMemoryType::Pageable;
  buffer.m_deviceId = descriptor.device_id;
  buffer.m_elementCount = storage->sizeBytes() / getElementSize(type);
  buffer.m_accelBuffer = std::move(storage);
  return buffer;
}

size_t TypedBuffer::getSizeBytes() const noexcept {
  // Backend storage is authoritative because its allocation policy is opaque.
  if (m_accelBuffer) {
    return m_accelBuffer->sizeBytes();
  }

  if (m_isExternalRef) {
    return m_elementCount * getElementSize(m_dataType);
  }
  return m_cpuData.size();
}

std::string_view TypedBuffer::backend() const noexcept {
  return m_accelBuffer ? m_accelBuffer->descriptor().backend
                       : std::string_view{"cpu"};
}

MemoryKind TypedBuffer::memoryKind() const noexcept {
  return m_accelBuffer ? m_accelBuffer->descriptor().kind : MemoryKind::Host;
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

const void *TypedBuffer::getRawHostPtr() const {
  return const_cast<TypedBuffer *>(this)->getRawHostPtr();
}

void *TypedBuffer::getRawHostPtr() {
  if (m_location != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to access Host pointer on Non-CPU buffer");
  }

  if (m_memoryType == BufferMemoryType::Pinned) {
    return m_accelBuffer ? m_accelBuffer->data() : nullptr;
  }

  if (m_isExternalRef) {
    return m_externalCpuPtr;
  }

  return m_cpuData.data();
}

void *TypedBuffer::getRawDevicePtr() const {
  if (m_location != BufferLocation::GpuDevice) {
    throw std::runtime_error(
        "Attempted to access Device pointer on Non-GPU buffer");
  }
  return m_accelBuffer ? m_accelBuffer->data() : nullptr;
}

void TypedBuffer::clear() { reset(); }

void TypedBuffer::resizeDiscard(size_t new_element_count) {
  if (new_element_count == m_elementCount && !m_isExternalRef) {
    return;
  }

  size_t new_size_bytes = new_element_count * getElementSize(m_dataType);

  if (m_accelBuffer) {
    m_accelBuffer = m_accelBuffer->allocate(new_size_bytes);
  } else {
    // CPU pageable. A wrapped external pointer is detached: this buffer is
    // being turned into an owned output buffer.
    m_isExternalRef = false;
    m_externalCpuPtr = nullptr;
    m_cpuData.assign(new_size_bytes, 0);
  }

  m_elementCount = new_element_count;
}

void TypedBuffer::resizePreserving(size_t new_element_count) {
  if (m_location != BufferLocation::CPU ||
      m_memoryType != BufferMemoryType::Pageable) {
    throw std::logic_error(
        "resizePreserving is only supported for CPU pageable buffers; use "
        "resizeDiscard (or copy explicitly) for pinned/GPU storage.");
  }

  size_t new_size_bytes = new_element_count * getElementSize(m_dataType);

  if (m_isExternalRef) {
    // Convert the wrapped memory into owned storage, keeping the overlap.
    std::vector<uint8_t> new_data(new_size_bytes, 0);
    if (m_externalCpuPtr && m_elementCount > 0) {
      size_t copy_size = std::min(getSizeBytes(), new_size_bytes);
      std::memcpy(new_data.data(), m_externalCpuPtr, copy_size);
    }
    m_isExternalRef = false;
    m_externalCpuPtr = nullptr;
    m_cpuData = std::move(new_data);
  } else {
    m_cpuData.resize(new_size_bytes);
  }

  m_elementCount = new_element_count;
}

} // namespace ai_core
