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
#include "ai_core/algo_data_types.hpp"
#include "registrar/infer_engine_registrar.hpp"
#include <logger.hpp>

namespace ai_core::dnn {
AlgoInferEngine::Impl::Impl(const std::string &moduleName,
                            const AlgoInferParams &inferParams)
    : moduleName_(moduleName), inferParams_(inferParams){};

InferErrorCode AlgoInferEngine::Impl::initialize() {
  try {
    AlgoConstructParams tempInferParams;
    tempInferParams.setParam("params", inferParams_);
    engine_ =
        InferEngineFactory::instance().create(moduleName_, tempInferParams);

    if (engine_ == nullptr) {
      LOG_ERRORS << "Failed to create inference engine for name: "
                 << inferParams_.name;
      return InferErrorCode::INIT_FAILED;
    }
  } catch (const std::exception &e) {
    LOG_ERRORS << "Failed to create inference engine: " << e.what();
    return InferErrorCode::INIT_FAILED;
  }

  if (engine_->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "Failed to initialize inference engine.";
    return InferErrorCode::INIT_FAILED;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoInferEngine::Impl::infer(const TensorData &modelInput,
                                            TensorData &modelOutput) {
  return engine_->infer(modelInput, modelOutput);
}

InferErrorCode AlgoInferEngine::Impl::terminate() {
  return engine_->terminate();
}

const ModelInfo &AlgoInferEngine::Impl::getModelInfo() const noexcept {
  if (engine_ == nullptr) {
    LOG_ERRORS << "Please initialize first";
    static ModelInfo modelInfo;
    return modelInfo;
  }
  return engine_->getModelInfo();
}
} // namespace ai_core::dnn
