/**
 * @file pinned_host_buffer_impl.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Abstract interface for pinned host memory implementation
 * @version 0.1
 * @date 2025-01-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_PINNED_HOST_BUFFER_IMPL_HPP
#define AI_CORE_PINNED_HOST_BUFFER_IMPL_HPP

#include <cstddef>
#include <memory>

namespace ai_core {

/**
 * @brief Abstract interface for pinned (page-locked) host memory
 *
 * This interface allows TypedBuffer to manage pinned memory without
 * directly depending on CUDA headers. The actual implementation
 * (CudaPinnedHostBufferImpl) lives in the CUDA module.
 *
 * Design Pattern: Bridge/pImpl
 * - TypedBuffer holds unique_ptr<PinnedHostBufferImpl>
 * - Concrete implementation uses cudaMallocHost/cudaFreeHost
 */
class PinnedHostBufferImpl {
public:
  virtual ~PinnedHostBufferImpl() = default;

  /**
   * @brief Get raw pointer to the pinned memory
   */
  virtual void *get() const = 0;

  /**
   * @brief Get size in bytes
   */
  virtual size_t getSizeBytes() const = 0;

  // ============================================================================
  // Static Factory Methods
  // ============================================================================

  /**
   * @brief Create a new pinned memory buffer
   *
   * @param sizeBytes Size to allocate
   * @return Unique pointer to implementation, or nullptr if CUDA unavailable
   *
   * @note On non-CUDA builds, this returns nullptr
   */
  static std::unique_ptr<PinnedHostBufferImpl> create(size_t sizeBytes);

  /**
   * @brief Clone an existing pinned buffer (deep copy)
   *
   * @param other Source buffer to copy from
   * @return New buffer with copied data
   */
  static std::unique_ptr<PinnedHostBufferImpl>
  clone(const PinnedHostBufferImpl &other);
};

} // namespace ai_core

#endif // AI_CORE_PINNED_HOST_BUFFER_IMPL_HPP