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

#include "ai_core/algo_inference.hpp"
#include "algo_infer_impl.hpp"

namespace ai_core::dnn {

AlgoInference::AlgoInference(const AlgoModuleTypes &module_types,
                             const AlgoInferParams &infer_params)
    : m_pImpl(std::make_unique<Impl>(module_types, infer_params)) {}

AlgoInference::~AlgoInference() = default;

InferErrorCode AlgoInference::initialize() { return m_pImpl->initialize(); }

InferErrorCode AlgoInference::infer(const AlgoInput &input,
                                    const AlgoPreprocParams &preproc_params,
                                    const AlgoPostprocParams &postproc_params,
                                    AlgoOutput &output) {
  return m_pImpl->infer(input, preproc_params, postproc_params, output);
}

InferErrorCode
AlgoInference::batchInfer(const std::vector<AlgoInput> &inputs,
                          const AlgoPreprocParams &preproc_params,
                          const AlgoPostprocParams &postproc_params,
                          std::vector<AlgoOutput> &outputs) {
  return m_pImpl->batchInfer(inputs, preproc_params, postproc_params, outputs);
}

InferErrorCode AlgoInference::terminate() { return m_pImpl->terminate(); }

const ModelInfo &AlgoInference::getModelInfo() const noexcept {
  return m_pImpl->getModelInfo();
}

const AlgoModuleTypes &AlgoInference::getModuleTypes() const noexcept {
  return m_pImpl->getModuleTypes();
}

} // namespace ai_core::dnn