/**
 * @file infer_config.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_INFER_PARAMS_TYPES_HPP
#define AI_CORE_INFER_PARAMS_TYPES_HPP

#include "ai_core/common_types.hpp"
#include <map>
#include <string>

namespace ai_core {
struct AlgoInferParams {
  std::string name;
  std::string model_path;
  bool need_decrypt = false;
  std::string decryptkey_str;
  DeviceType device_type;
  DataType data_type;
  std::map<std::string, size_t> max_output_buffer_sizes;
};
} // namespace ai_core

#endif