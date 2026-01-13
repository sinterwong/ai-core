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

#include "ai_core/i_infer_engine.hpp"
namespace ai_core::dnn {

void IInferEnginePlugin::prettyPrintModelInfos() {
  if (!m_modelInfo) {
    getModelInfo();
    if (!m_modelInfo) {
      return;
    }
  }
  std::cout << "Model Name: " << m_modelInfo->name << std::endl;
  std::cout << "Inputs:" << std::endl;
  for (const auto &input : m_modelInfo->inputs) {
    std::cout << "  Name: " << input.name << ", Shape: ";
    for (int64_t dim : input.shape) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Outputs:" << std::endl;
  for (const auto &output : m_modelInfo->outputs) {
    std::cout << "  Name: " << output.name << ", Shape: ";
    for (int64_t dim : output.shape) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }
}
}; // namespace ai_core::dnn
