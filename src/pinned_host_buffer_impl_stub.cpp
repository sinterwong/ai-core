/**
 * @file pinned_host_buffer_impl_stub.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-01-06
 *
 * @copyright Copyright (c) 2026
 *
 */

// 禁用TRT时编译
#ifndef WITH_TRT
#include "ai_core/pinned_host_buffer_impl.hpp"
#include <memory>
#include <stdexcept>

namespace ai_core {
class PinnedHostBufferImplStub : public PinnedHostBufferImpl {
public:
  // stub 不持有任何资源，所有需要实现的接口全部抛异常
  explicit PinnedHostBufferImplStub(size_t) {
    throw std::runtime_error(
        "Pinned memory support is not enabled in this build.");
  }

  ~PinnedHostBufferImplStub() override = default;

  void *get() const override { return nullptr; }
  size_t getSizeBytes() const override { return 0; }

  std::unique_ptr<PinnedHostBufferImpl> clone(const PinnedHostBufferImpl &) {
    throw std::runtime_error(
        "Pinned memory support is not enabled in this build.");
  }
};

std::unique_ptr<PinnedHostBufferImpl>
PinnedHostBufferImpl::create(size_t sizeBytes) {
  return std::make_unique<PinnedHostBufferImplStub>(sizeBytes);
}

std::unique_ptr<PinnedHostBufferImpl>
PinnedHostBufferImpl::clone(const PinnedHostBufferImpl &other) {
  return other.clone(other);
};
} // namespace ai_core
#endif
