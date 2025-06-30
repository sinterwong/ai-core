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
namespace ai_core {

enum class DeviceType { CPU = 0, GPU = 1 };

struct Shape {
  int w;
  int h;
  int c;
};

enum class DataType : u_char {
  FLOAT32 = 0,
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

} // namespace ai_core

#endif // __INFER_COMMON_TYPES_HPP__