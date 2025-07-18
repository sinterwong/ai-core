/**
 * @file algo_infer_engine.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __AI_CORE_ALGO_INFER_ENGINE_HPP__
#define __AI_CORE_ALGO_INFER_ENGINE_HPP__

#include "ai_core/infer_common_types.hpp"
#include "ai_core/infer_error_code.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoInferEngine {
public:
  AlgoInferEngine(const std::string &moduleName,
                  const AlgoInferParams &inferParams);

  ~AlgoInferEngine();

  InferErrorCode initialize();

  InferErrorCode infer(const TensorData &modelInput, TensorData &modelOutput);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};
} // namespace ai_core::dnn
#endif // __AI_CORE_ALGO_INFER_ENGINE_HPP__
