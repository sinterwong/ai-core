/**
 * @file typed_buffer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Type-safe buffer management with support for CPU, GPU, and Pinned
 * memory
 * @version 0.2
 * @date 2025-01-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_TYPED_BUFFER_HPP
#define AI_CORE_TYPED_BUFFER_HPP
#include "ai_core/device_buffer_impl.hpp"
#include "ai_core/infer_common_types.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ai_core {

/**
 * @brief Memory location classification
 */
enum class BufferLocation { CPU, GPU_DEVICE };

/**
 * @brief Memory type classification for CPU buffers
 *
 * This enum distinguishes between different allocation strategies
 * for host-side memory, which affects transfer performance.
 */
enum class BufferMemoryType {
  Pageable, ///< Standard pageable memory (malloc/new)
  Pinned,   ///< CUDA page-locked memory (cudaMallocHost)
  Managed   ///< CUDA managed memory (cudaMallocManaged) - future use
};

/**
 * @brief Forward declaration for pinned memory implementation
 *
 * This is implemented in the CUDA module to avoid CUDA headers in this file.
 */
class PinnedHostBufferImpl;

/**
 * @brief Type-safe buffer class supporting CPU, GPU, and Pinned memory
 *
 * TypedBuffer provides a unified interface for managing buffers across
 * different memory types and locations. It uses RAII for automatic
 * resource management.
 *
 * Memory Types:
 * - CPU Pageable: Standard heap memory, suitable for most CPU operations
 * - CPU Pinned: Page-locked memory for fast async GPU transfers
 * - GPU Device: CUDA device memory
 *
 * Usage Examples:
 * @code
 * // Standard CPU buffer
 * auto cpuBuf = TypedBuffer::createFromCpu(DataType::FLOAT32, data);
 *
 * // GPU buffer
 * auto gpuBuf = TypedBuffer::createFromGpu(DataType::FLOAT32, sizeBytes);
 *
 * // Pinned memory for async transfers (requires CUDA)
 * auto pinnedBuf = TypedBuffer::createPinnedHost(DataType::FLOAT32, sizeBytes);
 * @endcode
 */
class TypedBuffer {
public:
  TypedBuffer();
  ~TypedBuffer();

  TypedBuffer(const TypedBuffer &other);
  TypedBuffer &operator=(const TypedBuffer &other);
  TypedBuffer(TypedBuffer &&other) noexcept;
  TypedBuffer &operator=(TypedBuffer &&other) noexcept;

  // ============================================================================
  // Factory Methods - CPU Buffers
  // ============================================================================

  static TypedBuffer createFromCpu(DataType type,
                                   const std::vector<uint8_t> &data);
  static TypedBuffer createFromCpu(DataType type, std::vector<uint8_t> &&data);
  static TypedBuffer createFromCpuRef(DataType type, const void *hostPtr,
                                      size_t sizeBytes,
                                      bool manageMemory = false);

  // ============================================================================
  // Factory Methods - GPU Buffers
  // ============================================================================

  static TypedBuffer createFromGpu(DataType type, size_t sizeBytes,
                                   int deviceId = 0);
  static TypedBuffer createFromGpu(DataType type, void *devicePtr,
                                   size_t sizeBytes, int deviceId,
                                   bool manageMemory);

  // ============================================================================
  // Factory Methods - Pinned Memory
  // ============================================================================

  /**
   * @brief Create a pinned (page-locked) host memory buffer
   *
   * Pinned memory provides:
   * - Asynchronous H2D/D2H transfers
   * - Higher bandwidth via DMA
   * - Required for CUDA stream-based async operations
   *
   * @param type Data type for the buffer
   * @param sizeBytes Size in bytes to allocate
   * @return TypedBuffer backed by pinned memory
   *
   * @note This function requires CUDA runtime. On non-CUDA builds,
   *       it falls back to regular pageable memory.
   *
   * @warning Pinned memory allocation is expensive. Allocate once and reuse.
   *
   * @throws std::runtime_error if CUDA allocation fails
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

  /**
   * @brief Check if this buffer uses pinned memory
   */
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
  void setCpuData(DataType type, std::vector<uint8_t> &&data);

  void setGpuDataReference(DataType type, void *ptr, size_t sizeBytes,
                           int devId = 0);

  void resize(size_t newElementCount);

  void clear();

  static size_t getElementSize(DataType type) noexcept;

private:
  TypedBuffer(DataType type, const std::vector<uint8_t> &cpuData);
  TypedBuffer(DataType type, std::vector<uint8_t> &&cpuData);
  TypedBuffer(DataType type, size_t bufferSizeBytes, int deviceId);
  TypedBuffer(DataType type, void *devicePtr, size_t bufferSizeBytes,
              int deviceId, bool manageMemory);
  TypedBuffer(DataType type, const void *hostPtr, size_t sizeBytes,
              bool manageMemory);

  // Private constructor for pinned memory
  TypedBuffer(DataType type, size_t sizeBytes, BufferMemoryType memType);

  void reset();

  DataType mDataType{DataType::FLOAT32};
  BufferLocation mLocation{BufferLocation::CPU};
  BufferMemoryType mMemoryType{BufferMemoryType::Pageable};

  // CPU storage (pageable)
  std::vector<uint8_t> mCpuData;

  // External reference mode
  void *mExternalCpuPtr{nullptr};
  bool mIsExternalRef{false};
  bool mManageExternalCpu{false};

  // Pinned memory storage
  std::unique_ptr<PinnedHostBufferImpl> mPinnedImpl;

  // GPU storage
  std::unique_ptr<DeviceBufferImpl> mGpuImpl;
  size_t mDeviceBufferSizeBytes{0};
  int mDeviceId{0};

  size_t mElementCount{0};
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

  if (mPinnedImpl) {
    return reinterpret_cast<const T *>(getRawHostPtr());
  }
  if (mIsExternalRef && mExternalCpuPtr) {
    return reinterpret_cast<const T *>(mExternalCpuPtr);
  }
  return reinterpret_cast<const T *>(mCpuData.data());
}

template <typename T> T *TypedBuffer::getHostPtr() {
  if (mLocation != BufferLocation::CPU) {
    throw std::runtime_error(
        "Attempted to get host pointer from a non-CPU buffer.");
  }
  if (sizeof(T) != getElementSize(mDataType) && mElementCount > 0) {
    throw std::runtime_error("Mismatched type size for host data access.");
  }

  if (mPinnedImpl) {
    return reinterpret_cast<T *>(getRawHostPtr());
  }
  if (mIsExternalRef && mExternalCpuPtr) {
    return reinterpret_cast<T *>(mExternalCpuPtr);
  }
  return reinterpret_cast<T *>(mCpuData.data());
}

} // namespace ai_core

#endif // AI_CORE_TYPED_BUFFER_HPP