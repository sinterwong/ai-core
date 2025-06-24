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

#include "infer_common_types.hpp"
#include <string>

namespace infer {
struct InferParamBase {
  std::string name;
  std::string modelPath;
  bool needDecrypt = false;
  std::string decryptkeyStr;
  DeviceType deviceType;
  DataType dataType;
};

struct FrameInferParam : public InferParamBase {
  Shape inputShape;
};

} // namespace infer

#endif // __INFER_PARAMS_TYPES_HPP__