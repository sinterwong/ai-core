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
#ifndef AI_CORE_ALGO_INFER_ENGINE_HPP
#define AI_CORE_ALGO_INFER_ENGINE_HPP

#include "ai_core/infer_common_types.hpp"
#include "ai_core/infer_error_code.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoInferEngine {
public:
  AlgoInferEngine(const std::string &module_name,
                  const AlgoInferParams &infer_params);

  ~AlgoInferEngine();

  InferErrorCode initialize();

  InferErrorCode infer(const TensorData &model_input, TensorData &model_output);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
