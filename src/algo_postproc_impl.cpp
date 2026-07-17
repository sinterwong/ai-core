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
#include "ai_core/default_plugins.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/plugin_registrar.hpp"
#include "param_validation.hpp"

namespace ai_core::dnn {
AlgoPostproc::Impl::Impl(const std::string &module_name)
    : m_moduleName(module_name) {}

InferErrorCode
AlgoPostproc::Impl::initialize(const AlgoPostprocParams &postproc_params) {
  const auto validation = validateBoundParams(postproc_params);
  if (validation != InferErrorCode::SUCCESS) {
    return validation;
  }
  m_boundParams = postproc_params;

  registerDefaultPlugins();
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
                            std::shared_ptr<RuntimeContext> &runtime_context,
                            const AlgoPostprocParams *postproc_override) {
  if (m_postprocessor == nullptr) {
    LOG_ERROR_S << "Postprocessor is not initialized: " << m_moduleName;
    return InferErrorCode::NotInitialized;
  }
  const AlgoPostprocParams &params =
      postproc_override != nullptr ? *postproc_override : m_boundParams;
  try {
    const auto ret =
        m_postprocessor->process(model_output, params, output,
                                 runtime_context);
    if (ret != InferErrorCode::SUCCESS) {
      LOG_ERROR_S << "Failed to postprocess output.";
      return ret;
    }
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception in postprocessor '" << m_moduleName
                << "': " << e.what();
    return InferErrorCode::InferOutputError;
  } catch (...) {
    LOG_ERROR_S << "Unknown exception in postprocessor '" << m_moduleName
                << "'.";
    return InferErrorCode::InferOutputError;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPostproc::Impl::batchProcess(
    const TensorData &model_output, std::vector<AlgoOutput> &output,
    std::shared_ptr<RuntimeContext> &runtime_context,
    const AlgoPostprocParams *postproc_override) {
  if (m_postprocessor == nullptr) {
    LOG_ERROR_S << "Postprocessor is not initialized: " << m_moduleName;
    return InferErrorCode::NotInitialized;
  }
  const AlgoPostprocParams &params =
      postproc_override != nullptr ? *postproc_override : m_boundParams;
  try {
    const auto ret = m_postprocessor->batchProcess(model_output, params,
                                                   output, runtime_context);
    if (ret != InferErrorCode::SUCCESS) {
      LOG_ERROR_S << "Failed to batch postprocess output.";
      return ret;
    }
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception in postprocessor '" << m_moduleName
                << "': " << e.what();
    return InferErrorCode::InferOutputError;
  } catch (...) {
    LOG_ERROR_S << "Unknown exception in postprocessor '" << m_moduleName
                << "'.";
    return InferErrorCode::InferOutputError;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPostproc::Impl::terminate() {
  // No-op
  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn
