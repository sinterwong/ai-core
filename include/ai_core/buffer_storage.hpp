#ifndef AI_CORE_BUFFER_STORAGE_HPP
#define AI_CORE_BUFFER_STORAGE_HPP

#include <cstddef>
#include <memory>
#include <string_view>

namespace ai_core {

/** Physical memory domain reported by backend-owned storage. */
enum class MemoryKind { Host, HostPinned, Device, Unified };

/** Identifies the memory domain, backend, and device that own a buffer. */
struct MemoryDescriptor {
  MemoryKind kind{MemoryKind::Host};
  std::string_view backend{"cpu"};
  int device_id{0};
};

/**
 * @brief Polymorphic storage owned by `TypedBuffer`.
 *
 * CUDA, ROCm, OpenCL and future plugins implement allocation and copy policy;
 * ai-core never interprets a device pointer or calls a vendor runtime.
 */
class IBufferStorage {
public:
  virtual ~IBufferStorage() = default;

  virtual void *data() noexcept = 0;
  virtual const void *data() const noexcept = 0;
  virtual size_t sizeBytes() const noexcept = 0;
  virtual MemoryDescriptor descriptor() const noexcept = 0;

  /** Deep-copy the storage without changing its backend or memory domain. */
  virtual std::unique_ptr<IBufferStorage> clone() const = 0;

  /** Allocate uninitialized storage with the same backend and memory domain. */
  virtual std::unique_ptr<IBufferStorage> allocate(size_t size_bytes) const = 0;
};

} // namespace ai_core

#endif
