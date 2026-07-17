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
#include "ai_core/default_plugins.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/plugin_registrar.hpp"
#include "param_validation.hpp"

namespace ai_core::dnn {
AlgoPreproc::Impl::Impl(const std::string &module_name)
    : m_moduleName(module_name) {}

InferErrorCode
AlgoPreproc::Impl::initialize(const AlgoPreprocParams &preproc_params) {
  const auto validation = validateBoundParams(preproc_params);
  if (validation != InferErrorCode::SUCCESS) {
    return validation;
  }
  m_boundParams = preproc_params;

  registerDefaultPlugins();
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
    const AlgoInput &input, TensorData &model_input,
    std::shared_ptr<RuntimeContext> &runtime_context,
    const AlgoPreprocParams *preproc_override) {
  if (m_preprocessor == nullptr) {
    LOG_ERROR_S << "Preprocessor is not initialized: " << m_moduleName;
    return InferErrorCode::NotInitialized;
  }
  const AlgoPreprocParams &params =
      preproc_override != nullptr ? *preproc_override : m_boundParams;
  try {
    const auto ret =
        m_preprocessor->process(input, params, model_input, runtime_context);
    if (ret != InferErrorCode::SUCCESS) {
      LOG_ERROR_S << "Failed to preprocess input.";
      return ret;
    }
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception in preprocessor '" << m_moduleName
                << "': " << e.what();
    return InferErrorCode::InferPreprocessFailed;
  } catch (...) {
    LOG_ERROR_S << "Unknown exception in preprocessor '" << m_moduleName
                << "'.";
    return InferErrorCode::InferPreprocessFailed;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPreproc::Impl::batchProcess(
    const std::vector<AlgoInput> &input, TensorData &model_input,
    std::shared_ptr<RuntimeContext> &runtime_context,
    const AlgoPreprocParams *preproc_override) {
  if (m_preprocessor == nullptr) {
    LOG_ERROR_S << "Preprocessor is not initialized: " << m_moduleName;
    return InferErrorCode::NotInitialized;
  }
  const AlgoPreprocParams &params =
      preproc_override != nullptr ? *preproc_override : m_boundParams;
  try {
    const auto ret = m_preprocessor->batchProcess(input, params, model_input,
                                                  runtime_context);
    if (ret != InferErrorCode::SUCCESS) {
      LOG_ERROR_S << "Failed to batch preprocess input.";
      return ret;
    }
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception in preprocessor '" << m_moduleName
                << "': " << e.what();
    return InferErrorCode::InferPreprocessFailed;
  } catch (...) {
    LOG_ERROR_S << "Unknown exception in preprocessor '" << m_moduleName
                << "'.";
    return InferErrorCode::InferPreprocessFailed;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPreproc::Impl::terminate() {
  // No-op
  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn
