/**
 * @file algo_postproc.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "ai_core/algo_postprocessor.hpp"
#include "ai_core/tensor_data.hpp"
#include "algo_postproc_impl.hpp"

namespace ai_core::dnn {

AlgoPostproc::AlgoPostproc(const std::string &module_name)
    : m_pImpl(std::make_unique<Impl>(module_name)) {}

AlgoPostproc::~AlgoPostproc() = default;

InferErrorCode AlgoPostproc::initialize() { return m_pImpl->initialize(); }

InferErrorCode AlgoPostproc::process(
    const TensorData &model_output, const AlgoPostprocParams &postproc_params,
    AlgoOutput &output, std::shared_ptr<RuntimeContext> &runtime_context) {
  return m_pImpl->process(model_output, output, postproc_params,
                          runtime_context);
}

InferErrorCode
AlgoPostproc::batchProcess(const TensorData &model_output,
                           const AlgoPostprocParams &postproc_params,
                           std::vector<AlgoOutput> &output,
                           std::shared_ptr<RuntimeContext> &runtime_context) {
  return m_pImpl->batchProcess(model_output, output, postproc_params,
                               runtime_context);
}

InferErrorCode AlgoPostproc::terminate() { return m_pImpl->terminate(); }

} // namespace ai_core::dnn
