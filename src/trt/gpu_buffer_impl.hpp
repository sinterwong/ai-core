/**
 * @file gpu_buffer_impl.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <cstddef>
#include <memory>
namespace ai_core {
class GpuBufferImpl {
public:
  GpuBufferImpl(size_t sizeBytes);
  GpuBufferImpl(void *ptr, size_t sizeBytes, bool manageMemory);
  ~GpuBufferImpl();

  void *get() const;
  size_t getSizeBytes() const;

private:
  struct GpuBufferDeleter {
    void operator()(void *ptr) const;
  };

  std::shared_ptr<void> mManagedBuffer;
  void *mPtr{nullptr};
  size_t mSizeBytes{0};
};
} // namespace ai_core