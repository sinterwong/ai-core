/**
 * @file typed_buffer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __TYPED_BUFFER_HPP__
#define __TYPED_BUFFER_HPP__
#include "infer_common_types.hpp"
#include <cstdint>
#include <vector>

namespace infer {

struct TypedBuffer {
  DataType dataType;
  std::vector<uint8_t> data; // raw data
  size_t elementCount;

  template <typename T> const T *getTypedPtr() const {
    return reinterpret_cast<const T *>(data.data());
  }

  size_t getElementCount() const {
    size_t elemSize = getElementSize(dataType);
    return data.size() / elemSize;
  }

  static size_t getElementSize(DataType type) {
    switch (type) {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::FLOAT16:
      return sizeof(uint16_t);
    case DataType::INT8:
      return sizeof(int8_t);
    default:
      return 0;
    }
  }
};

} // namespace infer

#endif