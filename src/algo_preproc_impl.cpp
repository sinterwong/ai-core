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
AlgoPreproc::Impl::Impl(const std::string &moduleName,
                        const AlgoPreprocParams &preprocParams)
    : moduleName_(moduleName), preprocParams_(preprocParams) {}

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

InferErrorCode AlgoPreproc::Impl::process(AlgoInput &input,
                                          AlgoPreprocParams &preprocParams,
                                          TensorData &modelInput) {
  if (!preprocessor_->process(input, preprocParams, modelInput)) {
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
