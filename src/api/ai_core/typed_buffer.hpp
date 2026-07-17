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

#include "ai_core/i_accelerator_buffer.hpp"
#include "ai_core/common_types.hpp"

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
   * @brief Create a wrapper around existing CPU memory (Reference)
   * @param manageMemory If true, TypedBuffer will delete[] the pointer on
   * destruction
   */
  static TypedBuffer createFromCpuRef(DataType type, const void *host_ptr,
                                      size_t size_bytes,
                                      bool manage_memory = false);

  /**
   * @brief Create a GPU Device buffer
   */
  static TypedBuffer createFromGpu(DataType type, size_t size_bytes,
                                   int device_id = 0);

  /**
   * @brief Wrap existing GPU pointer
   */
  static TypedBuffer createFromGpu(DataType type, void *device_ptr,
                                   size_t size_bytes, int device_id,
                                   bool manage_memory);

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

  void setCpuData(DataType type, const std::vector<uint8_t> &data);
  void setGpuDataReference(DataType type, void *ptr, size_t size_bytes,
                           int dev_id = 0);

  /**
   * @brief Resize the buffer
   *
   * @note
   * - For CPU Pageable: Preserves data (std::vector resize behavior).
   * - For Pinned/GPU: Reallocates. Data preservation depends on backend
   * implementation (Current impl: destructive resize to avoid expensive
   * copies).
   */
  void resize(size_t new_element_count);

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

  // External Reference Storage
  void *m_externalCpuPtr{nullptr};
  bool m_isExternalRef{false};
  bool m_manageExternalCpu{false};

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