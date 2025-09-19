/**
 * @file algo_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ai_core/algo_infer.hpp"
#include "algo_infer_impl.hpp"

namespace ai_core::dnn {

AlgoInference::AlgoInference(const AlgoModuleTypes &moduleTypes,
                             const AlgoInferParams &inferParams)
    : pImpl(std::make_unique<Impl>(moduleTypes, inferParams)) {}

AlgoInference::~AlgoInference() = default;

InferErrorCode AlgoInference::initialize() { return pImpl->initialize(); }

InferErrorCode AlgoInference::infer(const AlgoInput &input,
                                    const AlgoPreprocParams &preprocParams,
                                    const AlgoPostprocParams &postprocParams,
                                    AlgoOutput &output) {
  return pImpl->infer(input, preprocParams, postprocParams, output);
}

InferErrorCode
AlgoInference::batchInfer(const std::vector<AlgoInput> &inputs,
                          const AlgoPreprocParams &preprocParams,
                          const AlgoPostprocParams &postprocParams,
                          std::vector<AlgoOutput> &outputs) {
  return pImpl->batchInfer(inputs, preprocParams, postprocParams, outputs);
}

InferErrorCode AlgoInference::terminate() { return pImpl->terminate(); }

const ModelInfo &AlgoInference::getModelInfo() const noexcept {
  return pImpl->getModelInfo();
}

const AlgoModuleTypes &AlgoInference::getModuleTypes() const noexcept {
  return pImpl->getModuleTypes();
}

} // namespace ai_core::dnn