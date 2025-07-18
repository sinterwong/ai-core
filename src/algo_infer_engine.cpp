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

#include "ai_core/algo_infer_engine.hpp"
#include "algo_infer_engine_impl.hpp"
#include <string>

namespace ai_core::dnn {

AlgoInferEngine::AlgoInferEngine(const std::string &moduleName,
                                 const AlgoInferParams &inferParams)
    : pImpl(std::make_unique<Impl>(moduleName, inferParams)) {}

AlgoInferEngine::~AlgoInferEngine() = default;

InferErrorCode AlgoInferEngine::initialize() { return pImpl->initialize(); }

InferErrorCode AlgoInferEngine::infer(const TensorData &modelInput,
                                      TensorData &modelOutput) {
  return pImpl->infer(modelInput, modelOutput);
}

InferErrorCode AlgoInferEngine::terminate() { return pImpl->terminate(); }

const ModelInfo &AlgoInferEngine::getModelInfo() const noexcept {
  return pImpl->getModelInfo();
}

} // namespace ai_core::dnn
