/**
 * @file typed_buffer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Type-safe buffer management with unified backend memory support
 * @version 0.3
 * @date 2026-01-06
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_TYPED_BUFFER_HPP
#define AI_CORE_TYPED_BUFFER_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/i_accelerator_buffer.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ai_core {

/**
 * @brief Memory location classification
 */
enum class BufferLocation { CPU, GpuDevice };

/**
 * @brief Memory type classification
 */
enum class BufferMemoryType {
  Pageable, // Standard pageable memory (std::vector)
  Pinned,   // Page-locked memory (Host RAM via Accelerator)
  Managed   // Unified Memory
};

/**
 * @brief Type-safe buffer class supporting CPU, GPU, and Pinned memory
 *
 * TypedBuffer provides a unified interface for managing buffers across
 * different memory types and locations. It uses RAII for automatic
 * resource management.
 *
 * Internal Architecture:
 * - CPU Pageable: Managed via std::vector<uint8_t>
 * - CPU External: Managed via raw pointers
 * - CPU Pinned / GPU Device: Managed via unified IAcceleratorBuffer
 */
class TypedBuffer {
public:
  TypedBuffer();
  ~TypedBuffer();

  // Rule of 5: Copy and Move semantics
  TypedBuffer(const TypedBuffer &other);
  TypedBuffer &operator=(const TypedBuffer &other);
  TypedBuffer(TypedBuffer &&other) noexcept;
  TypedBuffer &operator=(TypedBuffer &&other) noexcept;

  // ============================================================================
  // Factory Methods
  // ============================================================================

  /**
   * @brief Create from existing CPU data (Deep Copy)
   */
  static TypedBuffer createFromCpu(DataType type,
                                   const std::vector<uint8_t> &data);
  static TypedBuffer createFromCpu(DataType type, std::vector<uint8_t> &&data);

  /**
   * @brief Wrap existing CPU memory without taking ownership (like a view).
   * The caller keeps the memory alive for the buffer's lifetime. Copying the
   * TypedBuffer deep-copies into owned storage.
   */
  static TypedBuffer wrapCpu(DataType type, const void *host_ptr,
                             size_t size_bytes);

  /**
   * @brief Allocate a new GPU device buffer.
   */
  static TypedBuffer allocateGpu(DataType type, size_t size_bytes,
                                 int device_id = 0);

  /**
   * @brief Wrap an existing GPU pointer without taking ownership. The caller
   * keeps the allocation alive for the buffer's lifetime.
   */
  static TypedBuffer wrapGpu(DataType type, void *device_ptr,
                             size_t size_bytes, int device_id = 0);

  /**
   * @brief Create a Pinned (Page-locked) Host buffer
   *
   * Optimized for async H2D/D2H transfers.
   * Internally uses IAcceleratorBuffer with HostPinned type.
   */
  static TypedBuffer createPinnedHost(DataType type, size_t size_bytes);

  // ============================================================================
  // Property Queries
  // ============================================================================

  DataType dataType() const noexcept { return m_dataType; }
  BufferLocation location() const noexcept { return m_location; }
  BufferMemoryType memoryType() const noexcept { return m_memoryType; }

  size_t getSizeBytes() const noexcept;
  size_t getElementCount() const noexcept { return m_elementCount; }
  int getDeviceId() const noexcept;

  bool isPinned() const noexcept {
    return m_memoryType == BufferMemoryType::Pinned;
  }
  bool isReference() const noexcept { return m_isExternalRef; }

  // ============================================================================
  // Data Access - Host
  // ============================================================================

  template <typename T> const T *getHostPtr() const;
  template <typename T> T *getHostPtr();

  const void *getRawHostPtr() const;
  void *getRawHostPtr();

  // ============================================================================
  // Data Access - Device
  // ============================================================================

  void *getRawDevicePtr() const;

  // ============================================================================
  // Data Modification
  // ============================================================================

  /**
   * @brief Resize to `new_element_count`; contents are unspecified afterwards.
   *
   * Works for every memory type (pageable / pinned / GPU) - accelerator
   * buffers reallocate instead of copying. This is the right call for output
   * buffers that are about to be overwritten. A wrapped external buffer
   * (wrapCpu/wrapGpu) is detached and replaced by owned storage.
   */
  void resizeDiscard(size_t new_element_count);

  /**
   * @brief Resize preserving existing contents (std::vector semantics).
   *
   * Only supported for CPU pageable storage; throws std::logic_error for
   * pinned/GPU buffers, where a preserving resize would hide an expensive
   * device copy - do it explicitly if you need it.
   */
  void resizePreserving(size_t new_element_count);

  void clear();

  static size_t getElementSize(DataType type) noexcept;

private:
  // Private helper to reset state
  void reset();

  DataType m_dataType{DataType::FLOAT32};
  BufferLocation m_location{BufferLocation::CPU};
  BufferMemoryType m_memoryType{BufferMemoryType::Pageable};

  size_t m_elementCount{0};

  // Standard CPU Storage
  std::vector<uint8_t> m_cpuData;

  // External Reference Storage (non-owning)
  void *m_externalCpuPtr{nullptr};
  bool m_isExternalRef{false};

  // Unified Accelerator Storage (Handles both GPU VRAM and CPU Pinned RAM)
  std::unique_ptr<IAcceleratorBuffer> m_accelBuffer;

  int m_deviceId{0};
};

// ============================================================================
// Template Implementation
// ============================================================================

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