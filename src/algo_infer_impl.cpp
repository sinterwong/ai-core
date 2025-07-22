/**
 * @file algo_infer_impl.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <chrono>

#include "ai_core/infer_error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include "algo_infer_impl.hpp"
#include <logger.hpp>

namespace ai_core::dnn {
AlgoInference::Impl::Impl(const AlgoModuleTypes &algoModuleTypes,
                          const AlgoInferParams &inferParams)
    : algoModuleTypes_(algoModuleTypes), inferParams_(inferParams) {
  preprocessor_ = std::make_shared<AlgoPreproc>(algoModuleTypes_.preprocModule);
  engine_ = std::make_shared<AlgoInferEngine>(algoModuleTypes_.inferModule,
                                              inferParams_);
  postprocessor_ =
      std::make_shared<AlgoPostproc>(algoModuleTypes_.postprocModule);
};

InferErrorCode AlgoInference::Impl::initialize() {
  if (preprocessor_->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "Failed to initialize preprocessor.";
    return InferErrorCode::INIT_FAILED;
  }
  if (engine_->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "Failed to initialize inference engine.";
    return InferErrorCode::INIT_FAILED;
  }
  if (postprocessor_->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "Failed to initialize postprocessor.";
    return InferErrorCode::INIT_FAILED;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoInference::Impl::infer(AlgoInput &input,
                                          AlgoPreprocParams &preprocParams,
                                          AlgoOutput &output,
                                          AlgoPostprocParams &postprocParams) {
  if (engine_ == nullptr || preprocessor_ == nullptr ||
      postprocessor_ == nullptr) {
    LOG_ERRORS << "Please initialize first";
    return InferErrorCode::INIT_FAILED;
  }

  // prep const time
  auto startPre = std::chrono::steady_clock::now();
  TensorData modelInput;
  if (preprocessor_->process(input, preprocParams, modelInput) !=
      InferErrorCode::SUCCESS) {
    LOG_ERRORS << "Failed to preprocess input.";
    return InferErrorCode::INFER_PREPROCESS_FAILED;
  }
  auto endPre = std::chrono::steady_clock::now();
  auto durationPre =
      std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre);

  // infer cost time
  auto startInfer = std::chrono::steady_clock::now();
  TensorData modelOutput;
  auto ret = engine_->infer(modelInput, modelOutput);
  if (ret != InferErrorCode::SUCCESS) {
    return ret;
  }
  auto endInfer = std::chrono::steady_clock::now();
  auto durationInfer = std::chrono::duration_cast<std::chrono::milliseconds>(
      endInfer - startInfer);

  // post cost time
  auto startPost = std::chrono::steady_clock::now();
  if (postprocessor_->process(modelOutput, preprocParams, output,
                              postprocParams) != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "Failed to postprocess output.";
    return InferErrorCode::INFER_OUTPUT_ERROR;
  }
  auto endPost = std::chrono::steady_clock::now();
  auto durationPost = std::chrono::duration_cast<std::chrono::milliseconds>(
      endPost - startPost);

  LOG_INFOS << "Preprocess time: " << durationPre.count()
            << " ms, Infer time: " << durationInfer.count()
            << " ms, Postprocess time: " << durationPost.count() << " ms.";

  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoInference::Impl::terminate() {
  engine_->terminate();
  return InferErrorCode::SUCCESS;
}

const ModelInfo &AlgoInference::Impl::getModelInfo() const noexcept {
  if (engine_ == nullptr) {
    LOG_ERRORS << "Please initialize first";
    static ModelInfo modelInfo;
    return modelInfo;
  }
  return engine_->getModelInfo();
}

const AlgoModuleTypes &AlgoInference::Impl::getModuleTypes() const noexcept {
  return algoModuleTypes_;
};

} // namespace ai_core::dnn
