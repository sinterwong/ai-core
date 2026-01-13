/**
 * @file algo_preproc.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/tensor_data.hpp"
#include "algo_preproc_impl.hpp"

namespace ai_core::dnn {

AlgoPreproc::AlgoPreproc(const std::string &module_name)
    : m_pImpl(std::make_unique<Impl>(module_name)) {}

AlgoPreproc::~AlgoPreproc() = default;

InferErrorCode AlgoPreproc::initialize() { return m_pImpl->initialize(); }

InferErrorCode AlgoPreproc::process(
    const AlgoInput &input, const AlgoPreprocParams &preproc_params,
    TensorData &model_input, std::shared_ptr<RuntimeContext> &runtime_context) {
  return m_pImpl->process(input, preproc_params, model_input, runtime_context);
}

InferErrorCode AlgoPreproc::batchProcess(
    const std::vector<AlgoInput> &input, const AlgoPreprocParams &preproc_params,
    TensorData &model_input, std::shared_ptr<RuntimeContext> &runtime_context) {
  return m_pImpl->batchProcess(input, preproc_params, model_input, runtime_context);
}

InferErrorCode AlgoPreproc::terminate() { return m_pImpl->terminate(); }

} // namespace ai_core::dnn
