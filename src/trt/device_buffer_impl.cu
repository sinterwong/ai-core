#include "ai_core/device_buffer_impl.hpp"
#include "trt_utils.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace ai_core {

namespace trt_utils {
class DeviceBufferImplCuda : public DeviceBufferImpl {
private:
  struct GpuCudaBufferDeleter {
    void operator()(void *ptr) const {
      if (ptr) {
        CHECK_CUDA(cudaFree(ptr));
      }
    }
  };

  std::shared_ptr<void> mManagedBuffer;
  void *mPtr{nullptr};
  size_t mSizeBytes{0};

public:
  explicit DeviceBufferImplCuda(size_t sizeBytes) : mSizeBytes(sizeBytes) {
    void *ptr = nullptr;
    if (mSizeBytes > 0) {
      CHECK_CUDA(cudaMalloc(&ptr, mSizeBytes));
    }
    mManagedBuffer = std::shared_ptr<void>(ptr, GpuCudaBufferDeleter());
    mPtr = mManagedBuffer.get();
  }

  DeviceBufferImplCuda(void *ptr, size_t sizeBytes, bool manageMemory)
      : mPtr(ptr), mSizeBytes(sizeBytes) {
    if (manageMemory && ptr) {
      mManagedBuffer = std::shared_ptr<void>(ptr, GpuCudaBufferDeleter());
    }
  }

  DeviceBufferImplCuda(const DeviceBufferImplCuda &other)
      : mSizeBytes(other.mSizeBytes) {
    if (mSizeBytes > 0) {
      void *ptr = nullptr;
      CHECK_CUDA(cudaMalloc(&ptr, mSizeBytes));
      CHECK_CUDA(
          cudaMemcpy(ptr, other.mPtr, mSizeBytes, cudaMemcpyDeviceToDevice));
      mManagedBuffer = std::shared_ptr<void>(ptr, GpuCudaBufferDeleter());
      mPtr = mManagedBuffer.get();
    }
  }

  ~DeviceBufferImplCuda() override = default;

  void *get() const override { return mPtr; }
  size_t getSizeBytes() const override { return mSizeBytes; }
};
} // namespace trt_utils

std::unique_ptr<DeviceBufferImpl> DeviceBufferImpl::create(size_t sizeBytes) {
  return std::make_unique<trt_utils::DeviceBufferImplCuda>(sizeBytes);
}

std::unique_ptr<DeviceBufferImpl>
DeviceBufferImpl::create(void *ptr, size_t sizeBytes, bool manageMemory) {
  return std::make_unique<trt_utils::DeviceBufferImplCuda>(ptr, sizeBytes,
                                                           manageMemory);
}

std::unique_ptr<DeviceBufferImpl>
DeviceBufferImpl::clone(const DeviceBufferImpl &other) {
  const auto *cuda_impl =
      dynamic_cast<const trt_utils::DeviceBufferImplCuda *>(&other);
  if (!cuda_impl) {
    throw std::runtime_error("Cannot clone a non-CUDA implementation.");
  }
  return std::make_unique<trt_utils::DeviceBufferImplCuda>(*cuda_impl);
}
} // namespace ai_core