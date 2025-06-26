/**
 * @file algo_infer_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __AI_CORE_ALGO_INFER_BASE_HPP__
#define __AI_CORE_ALGO_INFER_BASE_HPP__

#include "types/algo_data_types.hpp"    // For AlgoInput, AlgoOutput
#include "types/infer_common_types.hpp" // For ModelInfo
#include "types/infer_error_code.hpp" // For InferErrorCode
#include <string>                       // For std::string

namespace ai_core::dnn {

class AlgoInferBase {
public:
  AlgoInferBase(){};
  virtual ~AlgoInferBase(){};

  virtual InferErrorCode initialize() = 0;

  virtual InferErrorCode infer(AlgoInput &input, AlgoOutput &output) = 0;

  virtual InferErrorCode terminate() = 0;

  virtual const ModelInfo &getModelInfo() const noexcept = 0;

  virtual const std::string &getModuleName() const noexcept = 0;
};
} // namespace ai_core::dnn
#endif // __AI_CORE_ALGO_INFER_BASE_HPP__
