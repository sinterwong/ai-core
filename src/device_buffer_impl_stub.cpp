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
#include "ai_core/accelerator_buffer_impl.hpp"
#include <memory>
#include <stdexcept>

namespace ai_core {
class AcceleratorBufferImplStub : public AcceleratorBufferImpl {
public:
  // stub 不持有任何资源，全部抛异常
  explicit AcceleratorBufferImplStub(size_t) {
    throw std::runtime_error("GPU/CUDA support is not enabled in this build.");
  }
  AcceleratorBufferImplStub(void *, size_t, bool) {
    throw std::runtime_error("GPU/CUDA support is not enabled in this build.");
  }

  ~AcceleratorBufferImplStub() override = default;

  void *get() const override { return nullptr; }
  size_t getSizeBytes() const override { return 0; }
};

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::create(size_t sizeBytes, AcceleratorMemoryType type) {
  throw std::runtime_error(
      "Cannot create GPU buffer: CUDA support is not compiled.");
}

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::createReference(void *ptr, size_t sizeBytes,
                                       AcceleratorMemoryType type,
                                       bool manageMemory) {
  throw std::runtime_error(
      "Cannot create GPU buffer: CUDA support is not compiled.");
}

std::unique_ptr<AcceleratorBufferImpl>
AcceleratorBufferImpl::clone(const AcceleratorBufferImpl &) {
  throw std::runtime_error(
      "Cannot clone GPU buffer: CUDA support is not compiled.");
}
} // namespace ai_core
#endif
