#ifndef AI_CORE_BUFFER_STORAGE_HPP
#define AI_CORE_BUFFER_STORAGE_HPP

#include <cstddef>
#include <memory>
#include <string_view>

namespace ai_core {

enum class MemoryKind { Host, HostPinned, Device, Unified };

struct MemoryDescriptor {
  MemoryKind kind{MemoryKind::Host};
  std::string_view backend{"cpu"};
  int device_id{0};
};

/**
 * Backend-owned storage carried by TypedBuffer.
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
  virtual std::unique_ptr<IBufferStorage> clone() const = 0;
  virtual std::unique_ptr<IBufferStorage>
  allocate(size_t size_bytes) const = 0;
};

} // namespace ai_core

#endif
