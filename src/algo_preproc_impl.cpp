/**
 * @file algo_preproc_impl.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "algo_preproc_impl.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/plugin_registrar.hpp"

namespace ai_core::dnn {
AlgoPreproc::Impl::Impl(const std::string &module_name)
    : m_moduleName(module_name) {}

InferErrorCode AlgoPreproc::Impl::initialize() {
  try {
    m_preprocessor =
        PreprocFactory::instance().create(m_moduleName, AlgoConstructParams{});
    if (m_preprocessor == nullptr) {
      LOG_ERROR_S << "Failed to create preprocessor for module: "
                  << m_moduleName;
      return InferErrorCode::InitFailed;
    }
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Failed to create preprocessor: " << e.what();
    return InferErrorCode::InitFailed;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPreproc::Impl::process(
    const AlgoInput &input, const AlgoPreprocParams &preproc_params,
    TensorData &model_input, std::shared_ptr<RuntimeContext> &runtime_context) {
  if (!m_preprocessor->process(input, preproc_params, model_input,
                               runtime_context)) {
    LOG_ERROR_S << "Failed to preprocess input.";
    return InferErrorCode::InferPreprocessFailed;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPreproc::Impl::batchProcess(
    const std::vector<AlgoInput> &input,
    const AlgoPreprocParams &preproc_params, TensorData &model_input,
    std::shared_ptr<RuntimeContext> &runtime_context) {
  if (!m_preprocessor->batchProcess(input, preproc_params, model_input,
                                    runtime_context)) {
    LOG_ERROR_S << "Failed to batch preprocess input.";
    return InferErrorCode::InferPreprocessFailed;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPreproc::Impl::terminate() {
  // No-op
  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn
