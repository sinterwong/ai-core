/**
 * @file algo_infer_engine_impl.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "algo_infer_engine_impl.hpp"
#include "ai_core/algo_types.hpp"
#include "ai_core/default_plugins.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/plugin_registrar.hpp"

namespace ai_core::dnn {
AlgoInferEngine::Impl::Impl(const std::string &module_name,
                            const AlgoInferParams &infer_params)
    : m_moduleName(module_name), m_inferParams(infer_params) {};

InferErrorCode AlgoInferEngine::Impl::initialize() {
  registerDefaultPlugins();
  try {
    AlgoConstructParams temp_infer_params;
    temp_infer_params.setParam("params", m_inferParams);
    m_engine =
        InferEngineFactory::instance().create(m_moduleName, temp_infer_params);

    if (m_engine == nullptr) {
      LOG_ERROR_S << "Failed to create inference engine for name: "
                  << m_inferParams.name;
      return InferErrorCode::InitFailed;
    }
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Failed to create inference engine: " << e.what();
    return InferErrorCode::InitFailed;
  }

  if (m_engine->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to initialize inference engine.";
    return InferErrorCode::InitFailed;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoInferEngine::Impl::infer(const TensorData &model_input,
                                            TensorData &model_output) {
  return m_engine->infer(model_input, model_output);
}

InferErrorCode AlgoInferEngine::Impl::terminate() {
  return m_engine->terminate();
}

const ModelInfo &AlgoInferEngine::Impl::getModelInfo() const noexcept {
  if (m_engine == nullptr) {
    LOG_ERROR_S << "Please initialize first";
    static ModelInfo model_info;
    return model_info;
  }
  return m_engine->getModelInfo();
}
} // namespace ai_core::dnn
