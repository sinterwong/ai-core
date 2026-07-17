/**
 * @file device_buffer_impl_stub.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-15
 *
 * @copyright Copyright (c) 2025
 *
 */

// 禁用TRT时编译
#ifndef WITH_TRT
#include "ai_core/i_accelerator_buffer.hpp"
#include <memory>
#include <stdexcept>

namespace ai_core {
class IAcceleratorBufferStub : public IAcceleratorBuffer {
public:
  // stub 不持有任何资源，全部抛异常
  explicit IAcceleratorBufferStub(size_t) {
    throw std::runtime_error("GPU/CUDA support is not enabled in this build.");
  }
  IAcceleratorBufferStub(void *, size_t, bool) {
    throw std::runtime_error("GPU/CUDA support is not enabled in this build.");
  }

  ~IAcceleratorBufferStub() override = default;

  void *get() const override { return nullptr; }
  size_t getSizeBytes() const override { return 0; }
};

std::unique_ptr<IAcceleratorBuffer>
IAcceleratorBuffer::create(size_t sizeBytes, AcceleratorMemoryType type) {
  throw std::runtime_error(
      "Cannot create GPU buffer: CUDA support is not compiled.");
}

std::unique_ptr<IAcceleratorBuffer>
IAcceleratorBuffer::createReference(void *ptr, size_t sizeBytes,
                                       AcceleratorMemoryType type,
                                       bool manageMemory) {
  throw std::runtime_error(
      "Cannot create GPU buffer: CUDA support is not compiled.");
}

std::unique_ptr<IAcceleratorBuffer>
IAcceleratorBuffer::clone(const IAcceleratorBuffer &) {
  throw std::runtime_error(
      "Cannot clone GPU buffer: CUDA support is not compiled.");
}
} // namespace ai_core
#endif
