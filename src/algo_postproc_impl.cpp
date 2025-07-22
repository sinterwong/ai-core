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
#include "logger.hpp"
#include "registrar/postproc_registrar.hpp"

namespace ai_core::dnn {
AlgoPostproc::Impl::Impl(const std::string &moduleName)
    : moduleName_(moduleName) {}

InferErrorCode AlgoPostproc::Impl::initialize() {
  try {
    postprocessor_ =
        PostprocFactory::instance().create(moduleName_, AlgoConstructParams{});

    if (postprocessor_ == nullptr) {
      LOG_ERRORS << "Failed to create postprocessor for module: "
                 << moduleName_;
      return InferErrorCode::INIT_FAILED;
    }
  } catch (const std::exception &e) {
    LOG_ERRORS << "Failed to create vision module: " << e.what();
    return InferErrorCode::INIT_FAILED;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPostproc::Impl::process(const TensorData &modelOutput,
                                           AlgoPreprocParams &preprocParams,
                                           AlgoOutput &output,
                                           AlgoPostprocParams &postprocParams) {
  if (!postprocessor_->process(modelOutput, preprocParams, output,
                               postprocParams)) {
    LOG_ERRORS << "Failed to postprocess output.";
    return InferErrorCode::INFER_OUTPUT_ERROR;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoPostproc::Impl::terminate() {
  // No-op
  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn
