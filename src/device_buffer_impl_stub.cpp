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
#include "ai_core/device_buffer_impl.hpp"
#include <memory>
#include <stdexcept>

namespace ai_core {
class DeviceBufferImplCudaStub : public DeviceBufferImpl {
public:
  // stub 不持有任何资源，全部抛异常
  explicit DeviceBufferImplCudaStub(size_t) {
    throw std::runtime_error("GPU/CUDA support is not enabled in this build.");
  }
  DeviceBufferImplCudaStub(void *, size_t, bool) {
    throw std::runtime_error("GPU/CUDA support is not enabled in this build.");
  }

  ~DeviceBufferImplCudaStub() override = default;

  void *get() const override { return nullptr; }
  size_t getSizeBytes() const override { return 0; }
};

std::unique_ptr<DeviceBufferImpl> DeviceBufferImpl::create(size_t) {
  throw std::runtime_error(
      "Cannot create GPU buffer: CUDA support is not compiled.");
}

std::unique_ptr<DeviceBufferImpl> DeviceBufferImpl::create(void *, size_t,
                                                           bool) {
  throw std::runtime_error(
      "Cannot create GPU buffer: CUDA support is not compiled.");
}

std::unique_ptr<DeviceBufferImpl>
DeviceBufferImpl::clone(const DeviceBufferImpl &) {
  throw std::runtime_error(
      "Cannot clone GPU buffer: CUDA support is not compiled.");
}
} // namespace ai_core
#endif
