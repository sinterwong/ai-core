/**
 * @file algo_infer_engine_impl.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef ALGO_INFER_ENGINE_IMPL_HPP
#define ALGO_INFER_ENGINE_IMPL_HPP

#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/infer_base.hpp"
#include "ai_core/infer_params_types.hpp"
#include <memory>
#include <string>

namespace ai_core::dnn {

class AlgoInferEngine::Impl {
public:
  Impl(const std::string &module_name, const AlgoInferParams &infer_params);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode infer(const TensorData &model_input, TensorData &model_output);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

private:
  std::string m_moduleName;
  AlgoInferParams m_inferParams;
  std::shared_ptr<IInferEnginePlugin> m_engine;
};
} // namespace ai_core::dnn
#endif
