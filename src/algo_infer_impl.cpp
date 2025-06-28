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

#include "ai_core/types/infer_error_code.hpp"
#include "algo_infer_impl.hpp"
#include "registrar/infer_engine_registrar.hpp"
#include "registrar/postproc_registrar.hpp"
#include "registrar/preproc_registrar.hpp"

#include "logger.hpp"

namespace ai_core::dnn {
AlgoInference::Impl::Impl(const AlgoModuleTypes &algoModuleTypes,
                          const AlgoInferParams &inferParams)
    : algoModuleTypes_(algoModuleTypes), inferParams_(inferParams){};

InferErrorCode AlgoInference::Impl::initialize() {
  try {
    AlgoConstructParams tempInferParams;
    tempInferParams.setParam("params", inferParams_);
    engine_ = InferEngineFactory::instance().create(
        algoModuleTypes_.inferModule, tempInferParams);

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

  try {
    AlgoConstructParams prepConsParams;
    preprocessor_ = PreprocFactory::instance().create(
        algoModuleTypes_.preprocModule, AlgoConstructParams{});
    if (preprocessor_ == nullptr) {
      LOG_ERRORS << "Failed to create preprocessor for module: "
                 << algoModuleTypes_.preprocModule;
      return InferErrorCode::INIT_FAILED;
    }
  } catch (const std::exception &e) {
    LOG_ERRORS << "Failed to create preprocessor: " << e.what();
    return InferErrorCode::INIT_FAILED;
  }

  try {
    postprocessor_ = PostprocFactory::instance().create(
        algoModuleTypes_.postprocModule, AlgoConstructParams{});

    if (postprocessor_ == nullptr) {
      LOG_ERRORS << "Failed to create postprocessor for module: "
                 << algoModuleTypes_.postprocModule;
      return InferErrorCode::INIT_FAILED;
    }
  } catch (const std::exception &e) {
    LOG_ERRORS << "Failed to create vision module: " << e.what();
    return InferErrorCode::INIT_FAILED;
  }

  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoInference::Impl::infer(AlgoInput &input,
                                          AlgoPreprocParams &preprocParams,
                                          AlgoOutput &output,
                                          AlgoPostprocParams &postprocParams) {
  if (engine_ == nullptr) {
    LOG_ERRORS << "Please initialize first";
    return InferErrorCode::INIT_FAILED;
  }

  // prep const time
  auto startPre = std::chrono::steady_clock::now();
  TensorData modelInput;
  if (!preprocessor_->process(input, preprocParams, modelInput)) {
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
  bool result = postprocessor_->process(modelOutput, preprocParams, output,
                                        postprocParams);
  auto endPost = std::chrono::steady_clock::now();
  auto durationPost = std::chrono::duration_cast<std::chrono::milliseconds>(
      endPost - startPost);

  LOG_INFOS << "Preprocess time: " << durationPre.count()
            << " ms, Infer time: " << durationInfer.count()
            << " ms, Postprocess time: " << durationPost.count() << " ms.";

  if (!result) {
    LOG_ERRORS << "Failed to postprocess output.";
    return InferErrorCode::INFER_OUTPUT_ERROR;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoInference::Impl::terminate() { return engine_->terminate(); }

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
