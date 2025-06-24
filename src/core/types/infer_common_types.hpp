/**
 * @file infer_common_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFER_COMMON_TYPES_HPP__
#define __INFER_COMMON_TYPES_HPP__

#include <string>
#include <vector>
namespace infer {

enum class DeviceType { CPU = 0, GPU = 1 };

struct Shape {
  int w;
  int h;
};

enum class DataType {
  FLOAT32,
  FLOAT16,
  INT8,
};

struct ModelInfo {
  std::string name;

  struct InputInfo {
    std::string name;
    std::vector<int64_t> shape;
  };

  struct OutputInfo {
    std::string name;
    std::vector<int64_t> shape;
  };

  std::vector<InputInfo> inputs;
  std::vector<OutputInfo> outputs;
};

} // namespace infer

#endif // __INFER_COMMON_TYPES_HPP__