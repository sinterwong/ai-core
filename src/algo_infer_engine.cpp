/**
 * @file algo_infer_engine.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ai_core/infer_engine_wrapper.hpp"
#include "algo_infer_engine_impl.hpp"
#include <string>

namespace ai_core::dnn {

AlgoInferEngine::AlgoInferEngine(const std::string &module_name,
                                 const AlgoInferParams &infer_params)
    : m_pImpl(std::make_unique<Impl>(module_name, infer_params)) {}

AlgoInferEngine::~AlgoInferEngine() = default;

InferErrorCode AlgoInferEngine::initialize() { return m_pImpl->initialize(); }

InferErrorCode AlgoInferEngine::infer(const TensorData &model_input,
                                      TensorData &model_output) {
  return m_pImpl->infer(model_input, model_output);
}

InferErrorCode AlgoInferEngine::terminate() { return m_pImpl->terminate(); }

const ModelInfo &AlgoInferEngine::getModelInfo() const noexcept {
  return m_pImpl->getModelInfo();
}

std::shared_ptr<IAsyncInferEngine> AlgoInferEngine::getAsyncEngine() const
    noexcept {
  return m_pImpl->getAsyncEngine();
}

} // namespace ai_core::dnn
