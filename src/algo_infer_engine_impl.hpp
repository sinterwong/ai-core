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
#ifndef __ALGO_INFER_ENGINE_IMPL_HPP__
#define __ALGO_INFER_ENGINE_IMPL_HPP__

#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/infer_base.hpp"
#include "ai_core/infer_params_types.hpp"
#include <memory>
#include <string>

namespace ai_core::dnn {

class AlgoInferEngine::Impl {
public:
  Impl(const std::string &moduleName, const AlgoInferParams &inferParams);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode infer(const TensorData &modelInput, TensorData &modelOutput);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

private:
  std::string moduleName_;
  AlgoInferParams inferParams_;
  std::shared_ptr<InferBase> engine_;
};
} // namespace ai_core::dnn
#endif
