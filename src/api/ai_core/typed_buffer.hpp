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

#include "ai_core/accelerator_buffer_impl.hpp"
#include "ai_core/infer_common_types.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ai_core {

/**
 * @brief Memory location classification
 */
enum class BufferLocation { CPU, GPU_DEVICE };

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
 * - CPU Pinned / GPU Device: Managed via unified AcceleratorBufferImpl
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
  static TypedBuffer createFromCpuRef(DataType type, const void *hostPtr,
                                      size_t sizeBytes,
                                      bool manageMemory = false);

  /**
   * @brief Create a GPU Device buffer
   */
  static TypedBuffer createFromGpu(DataType type, size_t sizeBytes,
                                   int deviceId = 0);

  /**
   * @brief Wrap existing GPU pointer
   */
  static TypedBuffer createFromGpu(DataType type, void *devicePtr,
                                   size_t sizeBytes, int deviceId,
                                   bool manageMemory);

  /**
   * @brief Create a Pinned (Page-locked) Host buffer
   *
   * Optimized for async H2D/D2H transfers.
   * Internally uses AcceleratorBufferImpl with HostPinned type.
   */
  static TypedBuffer createPinnedHost(DataType type, size_t sizeBytes);

  // ============================================================================
  // Property Queries
  // ============================================================================

  DataType dataType() const noexcept { return mDataType; }
  BufferLocation location() const noexcept { return mLocation; }
  BufferMemoryType memoryType() const noexcept { return mMemoryType; }

  size_t getSizeBytes() const noexcept;
  size_t getElementCount() const noexcept { return mElementCount; }
  int getDeviceId() const noexcept;

  bool isPinned() const noexcept {
    return mMemoryType == BufferMemoryType::Pinned;
  }
  bool isReference() const noexcept { return mIsExternalRef; }

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
  void setGpuDataReference(DataType type, void *ptr, size_t sizeBytes,
                           int devId = 0);

  /**
   * @brief Resize the buffer
   *
   * @note
   * - For CPU Pageable: Preserves data (std::vector resize behavior).
   * - For Pinned/GPU: Reallocates. Data preservation depends on backend
   * implementation (Current impl: destructive resize to avoid expensive
   * copies).
   */
  void resize(size_t newElementCount);

  void clear();

  static size_t getElementSize(DataType type) noexcept;

private:
  // Private helper to reset state
  void reset();

  DataType mDataType{DataType::FLOAT32};
  BufferLocation mLocation{BufferLocation::CPU};
  BufferMemoryType mMemoryType{BufferMemoryType::Pageable};

  size_t mElementCount{0};

  // Standard CPU Storage
  std::vector<uint8_t> mCpuData;

  // External Reference Storage
  void *mExternalCpuPtr{nullptr};
  bool mIsExternalRef{false};
  bool mManageExternalCpu{false};

  // Unified Accelerator Storage (Handles both GPU VRAM and CPU Pinned RAM)
  std::unique_ptr<AcceleratorBufferImpl> mAccelBuffer;

  int mDeviceId{0};
};

// ============================================================================
// Template Implementation
// ============================================================================

template <typename T> const T *TypedBuffer::getHostPtr() const {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  if (sizeof(T) != getElementSize(mDataType) && mElementCount > 0) {
    throw std::runtime_error("Mismatched type size for host data access.");
  }
  return reinterpret_cast<const T *>(getRawHostPtr());
}

template <typename T> T *TypedBuffer::getHostPtr() {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  if (sizeof(T) != getElementSize(mDataType) && mElementCount > 0) {
    throw std::runtime_error("Mismatched type size for host data access.");
  }
  return reinterpret_cast<T *>(getRawHostPtr());
}

} // namespace ai_core

#endif // AI_CORE_TYPED_BUFFER_HPP