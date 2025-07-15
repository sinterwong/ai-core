/**
 * @file device_buffer_impl.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <memory>

namespace ai_core {
class DeviceBufferImpl {
public:
  virtual ~DeviceBufferImpl() = default;
  virtual void *get() const = 0;
  virtual size_t getSizeBytes() const = 0;

  // static factory
  static std::unique_ptr<DeviceBufferImpl> create(size_t sizeBytes);
  static std::unique_ptr<DeviceBufferImpl> create(void *ptr, size_t sizeBytes,
                                                  bool manageMemory);

  // clone factory
  static std::unique_ptr<DeviceBufferImpl> clone(const DeviceBufferImpl &other);
};
} // namespace ai_core