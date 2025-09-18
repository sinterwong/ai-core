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
#include "registrar/preproc_registrar.hpp"
#include <logger.hpp>

namespace ai_core::dnn {
AlgoPreproc::Impl::Impl(const std::string &moduleName)
    : moduleName_(moduleName) {}

InferErrorCode AlgoPreproc::Impl::initialize() {
  try {
    preprocessor_ =
        PreprocFactory::instance().create(moduleName_, AlgoConstructParams{});
    if (preprocessor_ == nullptr) {
      LOG_ERRORS << "Failed to create preprocessor for module: " << moduleName_;
      return InferErrorCode::INIT_FAILED;
    }
  } catch (const std::exception &e) {
    LOG_ERRORS << "Failed to create preprocessor: " << e.what();
    return InferErrorCode::INIT_FAILED;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPreproc::Impl::process(
    const AlgoInput &input, const AlgoPreprocParams &preprocParams,
    TensorData &modelInput, std::shared_ptr<RuntimeContext> &runtimeContext) {
  if (!preprocessor_->process(input, preprocParams, modelInput,
                              runtimeContext)) {
    LOG_ERRORS << "Failed to preprocess input.";
    return InferErrorCode::INFER_PREPROCESS_FAILED;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPreproc::Impl::terminate() {
  // No-op
  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn
