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

InferErrorCode
AlgoInference::initialize(const AlgoPreprocParams &preproc_params,
                          const AlgoPostprocParams &postproc_params) {
  return m_pImpl->initialize(preproc_params, postproc_params);
}

InferErrorCode AlgoInference::infer(const AlgoInput &input, AlgoOutput &output,
                                    const AlgoPreprocParams *preproc_override,
                                    const AlgoPostprocParams *postproc_override) {
  return m_pImpl->infer(input, output, preproc_override, postproc_override);
}

InferErrorCode
AlgoInference::batchInfer(const std::vector<AlgoInput> &inputs,
                          std::vector<AlgoOutput> &outputs,
                          const AlgoPreprocParams *preproc_override,
                          const AlgoPostprocParams *postproc_override) {
  return m_pImpl->batchInfer(inputs, outputs, preproc_override,
                             postproc_override);
}

InferErrorCode AlgoInference::terminate() { return m_pImpl->terminate(); }

const ModelInfo &AlgoInference::getModelInfo() const noexcept {
  return m_pImpl->getModelInfo();
}

const AlgoModuleTypes &AlgoInference::getModuleTypes() const noexcept {
  return m_pImpl->getModuleTypes();
}

std::shared_ptr<IAsyncInferEngine> AlgoInference::getAsyncEngine() const
    noexcept {
  return m_pImpl->getAsyncEngine();
}

} // namespace ai_core::dnn