/**
 * @file infer_base.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <iostream>

#include "infer_base.hpp"
namespace ai_core::dnn {

void InferBase::prettyPrintModelInfos() {
  if (!modelInfo) {
    getModelInfo();
    if (!modelInfo) {
      return;
    }
  }
  std::cout << "Model Name: " << modelInfo->name << std::endl;
  std::cout << "Inputs:" << std::endl;
  for (const auto &input : modelInfo->inputs) {
    std::cout << "  Name: " << input.name << ", Shape: ";
    for (int64_t dim : input.shape) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Outputs:" << std::endl;
  for (const auto &output : modelInfo->outputs) {
    std::cout << "  Name: " << output.name << ", Shape: ";
    for (int64_t dim : output.shape) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }
}
}; // namespace ai_core::dnn
