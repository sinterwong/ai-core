/**
 * @file algo_postproc_impl.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "algo_postproc_impl.hpp"
#include "ai_core/plugin_registrar.hpp"
#include "ai_core/logger.hpp"

namespace ai_core::dnn {
AlgoPostproc::Impl::Impl(const std::string &module_name)
    : m_moduleName(module_name) {}

InferErrorCode AlgoPostproc::Impl::initialize() {
  try {
    m_postprocessor =
        PostprocFactory::instance().create(m_moduleName, AlgoConstructParams{});

    if (m_postprocessor == nullptr) {
      LOG_ERROR_S << "Failed to create postprocessor for module: "
                  << m_moduleName;
      return InferErrorCode::InitFailed;
    }
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Failed to create vision module: " << e.what();
    return InferErrorCode::InitFailed;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode
AlgoPostproc::Impl::process(const TensorData &model_output, AlgoOutput &output,
                            const AlgoPostprocParams &postproc_params,
                            std::shared_ptr<RuntimeContext> &runtime_context) {
  if (!m_postprocessor->process(model_output, postproc_params, output,
                               runtime_context)) {
    LOG_ERROR_S << "Failed to postprocess output.";
    return InferErrorCode::InferOutputError;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPostproc::Impl::batchProcess(
    const TensorData &model_output, std::vector<AlgoOutput> &output,
    const AlgoPostprocParams &postproc_params,
    std::shared_ptr<RuntimeContext> &runtime_context) {
  if (!m_postprocessor->batchProcess(model_output, postproc_params, output,
                                    runtime_context)) {
    LOG_ERROR_S << "Failed to batch postprocess output.";
    return InferErrorCode::InferOutputError;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPostproc::Impl::terminate() {
  // No-op
  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn
