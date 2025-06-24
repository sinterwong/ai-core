/**
 * @file vision_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "vision_infer.hpp"
#include "logger/logger.hpp"
#include "vision_registrar.hpp"

namespace infer::dnn::vision {
VisionInfer::VisionInfer(const std::string &moduleName,
                         const AlgoInferParams &inferParam,
                         const AlgoPostprocParams &postproc)
    : moduleName(moduleName), inferParams(inferParam),
      postprocParams(postproc){};

InferErrorCode VisionInfer::initialize() {
  auto frameInferParams = inferParams.getParams<FrameInferParam>();
  if (frameInferParams == nullptr) {
    LOG_ERRORS << "frameInferParams is nullptr";
    return InferErrorCode::INIT_FAILED;
  }

  engine = std::make_shared<FrameInference>(*frameInferParams);

  AlgoConstructParams params;
  params.params = {{"params", postprocParams}};

  try {
    vision = VisionFactory::instance().create(moduleName, params);
  } catch (const std::exception &e) {
    LOG_ERRORS << "Failed to create vision module: " << e.what();
    return InferErrorCode::INIT_FAILED;
  }

  return engine->initialize();
}

InferErrorCode VisionInfer::infer(AlgoInput &input, AlgoOutput &output) {
  if (engine == nullptr) {
    LOG_ERRORS << "Please initialize first";
    return InferErrorCode::INIT_FAILED;
  }

  ModelOutput modelOutput;
  auto ret = engine->infer(input, modelOutput);
  if (ret != InferErrorCode::SUCCESS) {
    return ret;
  }

  auto frameInput = input.getParams<FrameInput>();
  if (frameInput == nullptr) {
    LOG_ERRORS << "frameInput is nullptr";
    return InferErrorCode::INFER_FAILED;
  }

  // post cost time
  auto startPost = std::chrono::steady_clock::now();
  bool result = vision->processOutput(modelOutput, frameInput->args, output);
  auto endPost = std::chrono::steady_clock::now();
  auto durationPost = std::chrono::duration_cast<std::chrono::milliseconds>(
      endPost - startPost);

  auto frameInferParams = inferParams.getParams<FrameInferParam>();
  if (frameInferParams == nullptr) {
    LOG_ERRORS << "frameInferParams is nullptr";
    return InferErrorCode::INIT_FAILED;
  }
  LOG_INFOS << frameInferParams->name << " postprocess cost "
            << durationPost.count() << " ms";
  return result ? InferErrorCode::SUCCESS : InferErrorCode::INFER_FAILED;
}

InferErrorCode VisionInfer::terminate() { return engine->terminate(); }

const ModelInfo &VisionInfer::getModelInfo() const noexcept {
  if (engine == nullptr) {
    LOG_ERRORS << "Please initialize first";
    static ModelInfo modelInfo;
    return modelInfo;
  }
  return engine->getModelInfo();
}

const std::string &VisionInfer::getModuleName() const noexcept {
  return moduleName;
};

} // namespace infer::dnn::vision
