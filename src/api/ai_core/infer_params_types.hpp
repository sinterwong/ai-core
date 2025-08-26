/**
 * @file infer_params_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __INFER_PARAMS_TYPES_HPP__
#define __INFER_PARAMS_TYPES_HPP__

#include "ai_core/infer_common_types.hpp"
#include <map>
#include <string>

namespace ai_core {
struct AlgoInferParams {
  std::string name;
  std::string modelPath;
  bool needDecrypt = false;
  std::string decryptkeyStr;
  DeviceType deviceType;
  DataType dataType;
  std::map<std::string, size_t> maxOutputBufferSizes;
};
} // namespace ai_core

#endif // __INFER_PARAMS_TYPES_HPP__