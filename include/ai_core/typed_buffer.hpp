#ifndef AI_CORE_TYPED_BUFFER_HPP
#define AI_CORE_TYPED_BUFFER_HPP

#include "ai_core/buffer_storage.hpp"
#include "ai_core/common_types.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ai_core {

/** Whether a buffer is host-accessible or device-only. */
enum class BufferLocation { CPU, GpuDevice };

/** Allocation policy used by a `TypedBuffer`. */
enum class BufferMemoryType {
  Pageable, ///< Ordinary pageable host memory.
  Pinned,   ///< Page-locked host memory allocated by a backend.
  Managed   ///< Unified memory accessible from host and device.
};

/**
 * @brief Owning or non-owning tensor buffer with element-type metadata.
 *
 * Pageable host data is stored directly. Pinned, unified, and device storage
 * is owned through `IBufferStorage`, leaving vendor runtimes outside the core
 * API. Copying always produces independent owned storage, including when the
 * source wraps external memory.
 *
 * @par Thread safety
 * Value type with no internal synchronization. Concurrent const access (e.g.
 * multiple readers of getHostPtr) is safe; any concurrent mutation, or mutation
 * racing a read, requires external synchronization.
 */
class TypedBuffer {
public:
  TypedBuffer();
  ~TypedBuffer();

  TypedBuffer(const TypedBuffer &other);
  TypedBuffer &operator=(const TypedBuffer &other);
  TypedBuffer(TypedBuffer &&other) noexcept;
  TypedBuffer &operator=(TypedBuffer &&other) noexcept;

  /** Copy `data` into owned pageable host storage. */
  static TypedBuffer createFromCpu(DataType type,
                                   const std::vector<uint8_t> &data);

  /** Move `data` into owned pageable host storage. */
  static TypedBuffer createFromCpu(DataType type, std::vector<uint8_t> &&data);

  /**
   * @brief Wrap external host memory without taking ownership.
   *
   * `host_ptr` must remain valid for this buffer's lifetime. Copying the
   * buffer deep-copies the bytes into owned pageable storage.
   */
  static TypedBuffer wrapCpu(DataType type, const void *host_ptr,
                             size_t size_bytes);

  /**
   * @brief Take ownership of storage supplied by a backend plugin.
   * @throws std::invalid_argument If `storage` is null.
   */
  static TypedBuffer fromStorage(DataType type,
                                 std::unique_ptr<IBufferStorage> storage);

  DataType dataType() const noexcept { return m_dataType; }
  BufferLocation location() const noexcept { return m_location; }
  BufferMemoryType memoryType() const noexcept { return m_memoryType; }

  size_t getSizeBytes() const noexcept;
  size_t getElementCount() const noexcept { return m_elementCount; }
  int getDeviceId() const noexcept;
  std::string_view backend() const noexcept;
  MemoryKind memoryKind() const noexcept;

  bool isPinned() const noexcept {
    return m_memoryType == BufferMemoryType::Pinned;
  }
  bool isReference() const noexcept { return m_isExternalRef; }

  /**
   * @brief Return a typed host pointer without copying.
   *
   * `sizeof(T)` must match the buffer's declared `DataType`; callers remain
   * responsible for choosing the corresponding C++ type.
   *
   * @throws std::runtime_error If the buffer is device-only or sizes differ.
   */
  template <typename T> const T *getHostPtr() const;
  template <typename T> T *getHostPtr();

  const void *getRawHostPtr() const;
  void *getRawHostPtr();

  /**
   * @brief Return the backend-owned device pointer without copying.
   * @throws std::runtime_error If the buffer is host-accessible.
   */
  void *getRawDevicePtr() const;

  /**
   * @brief Resize to `new_element_count`; contents are unspecified afterwards.
   *
   * Works for every memory type (pageable, pinned, unified, or device);
   * buffers reallocate through their owning plugin. This is the right call for
   * output buffers that are about to be overwritten.
   */
  void resizeDiscard(size_t new_element_count);

  /**
   * @brief Resize preserving existing contents (std::vector semantics).
   *
   * Only supported for pageable host storage. A wrapped external buffer is
   * detached into owned storage before resizing.
   *
   * @throws std::logic_error For pinned, unified, or device storage.
   */
  void resizePreserving(size_t new_element_count);

  void clear();

  static size_t getElementSize(DataType type) noexcept;

private:
  void reset();

  DataType m_dataType{DataType::FLOAT32};
  BufferLocation m_location{BufferLocation::CPU};
  BufferMemoryType m_memoryType{BufferMemoryType::Pageable};

  size_t m_elementCount{0};

  std::vector<uint8_t> m_cpuData;

  // This pointer is never freed; copying it materializes owned storage.
  void *m_externalCpuPtr{nullptr};
  bool m_isExternalRef{false};

  // Vendor-specific allocation and copy behavior remains behind this interface.
  std::unique_ptr<IBufferStorage> m_accelBuffer;

  int m_deviceId{0};
};

// Template definitions

template <typename T> const T *TypedBuffer::getHostPtr() const {
  if (m_location != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  if (sizeof(T) != getElementSize(m_dataType) && m_elementCount > 0) {
    throw std::runtime_error("Mismatched type size for host data access.");
  }
  return reinterpret_cast<const T *>(getRawHostPtr());
}

template <typename T> T *TypedBuffer::getHostPtr() {
  if (m_location != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  if (sizeof(T) != getElementSize(m_dataType) && m_elementCount > 0) {
    throw std::runtime_error("Mismatched type size for host data access.");
  }
  return reinterpret_cast<T *>(getRawHostPtr());
}

} // namespace ai_core

#endif // AI_CORE_TYPED_BUFFER_HPP
