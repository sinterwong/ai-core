/**
 * @file accelerator_buffer_impl.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Unified interface for backend-managed memory (Device & Pinned)
 * @version 0.2
 * @date 2026-01-06
 */

#ifndef AI_CORE_ACCELERATOR_BUFFER_IMPL_HPP
#define AI_CORE_ACCELERATOR_BUFFER_IMPL_HPP
#include <memory>

namespace ai_core {

/**
 * @brief Enum defining the type of accelerator memory
 */
enum class AcceleratorMemoryType {
  Device,     // GPU VRAM (cudaMalloc)
  HostPinned, // CPU RAM, Page-Locked (cudaMallocHost)
  Managed     // Unified Memory (cudaMallocManaged) - Optional
};

/**
 * @brief Abstract interface for memory managed by the acceleration backend
 *
 * This unifies handling for:
 * 1. Device Memory (VRAM)
 * 2. Pinned Host Memory (RAM registered with Driver)
 */
class AcceleratorBufferImpl {
public:
  virtual ~AcceleratorBufferImpl() = default;

  virtual void *get() const = 0;
  virtual size_t getSizeBytes() const = 0;
  virtual AcceleratorMemoryType getType() const = 0;

  /**
   * @brief Factory method to create specific memory type
   */
  static std::unique_ptr<AcceleratorBufferImpl>
  create(size_t sizeBytes, AcceleratorMemoryType type);

  /**
   * @brief Factory wrapper for wrapping existing pointers (advanced use)
   */
  static std::unique_ptr<AcceleratorBufferImpl>
  createReference(void *ptr, size_t sizeBytes, AcceleratorMemoryType type,
                  bool manageMemory);

  /**
   * @brief Clone factory
   */
  static std::unique_ptr<AcceleratorBufferImpl>
  clone(const AcceleratorBufferImpl &other);
};

} // namespace ai_core
#endif