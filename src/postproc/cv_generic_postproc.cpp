/**
 * @file cv_generic_postproc.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "cv_generic_postproc.hpp"
#include "postproc/fpr_cls.hpp"
#include "postproc/ocr_reco.hpp"
#include "postproc/raw_feature.hpp"
#include "postproc/softmax_cls.hpp"
#include "postproc/unet_daul_out_seg.hpp"
#include <logger.hpp>
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool CVGenericPostproc::process(
    const TensorData &modelOutput, const AlgoPostprocParams &postArgs,
    AlgoOutput &algoOutput,
    std::shared_ptr<RuntimeContext> &runtimeContext) const {

  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  auto params = postArgs.getParams<GenericPostParams>();
  if (params == nullptr) {
    LOG_ERRORS << "GenericPostParams params is nullptr";
    throw std::runtime_error("GenericPostParams params is nullptr");
  }

  switch (params->algoType) {
  case GenericPostParams::AlgoType::SOFTMAX_CLS: {
    FrameTransformContext prepRuntimeArgs;
    SoftmaxCls postproc;
    return postproc.process(modelOutput, prepRuntimeArgs, *params, algoOutput);
  }
  case GenericPostParams::AlgoType::FPR_CLS: {
    FrameTransformContext prepRuntimeArgs;
    FprCls postproc;
    return postproc.process(modelOutput, prepRuntimeArgs, *params, algoOutput);
  }
  case GenericPostParams::AlgoType::RAW_FEATURE: {
    FrameTransformContext prepRuntimeArgs;
    RawFeature postproc;
    return postproc.process(modelOutput, prepRuntimeArgs, *params, algoOutput);
  }
  case GenericPostParams::AlgoType::UNET_DUAL_OUTPUT: {
    if (!runtimeContext->has<FrameTransformContext>("preproc_runtime_args")) {
      LOG_ERRORS << "FramePreprocessArg is nullptr";
      throw std::runtime_error("FramePreprocessArg is nullptr");
    }
    const auto &prepRuntimeArgs =
        runtimeContext->getParam<FrameTransformContext>("preproc_runtime_args");
    UNetDaulOutputSeg postproc;
    return postproc.process(modelOutput, prepRuntimeArgs, *params, algoOutput);
  }
  case GenericPostParams::AlgoType::OCR_RECO: {
    FrameTransformContext prepRuntimeArgs;
    OCRReco postproc;
    return postproc.process(modelOutput, prepRuntimeArgs, *params, algoOutput);
  }
  default: {
    LOG_ERRORS << "Unknown generic algorithm type: "
               << static_cast<int>(params->algoType);
    return false;
  }
  }
  return true;
}

bool CVGenericPostproc::batchProcess(
    const TensorData &modelOutput, const AlgoPostprocParams &postArgs,
    std::vector<AlgoOutput> &output,
    std::shared_ptr<RuntimeContext> &runtimeContext) const {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  auto params = postArgs.getParams<GenericPostParams>();
  if (params == nullptr) {
    LOG_ERRORS << "GenericPostParams params is nullptr";
    throw std::runtime_error("GenericPostParams params is nullptr");
  }

  switch (params->algoType) {
  case GenericPostParams::AlgoType::SOFTMAX_CLS: {
    std::vector<FrameTransformContext> prepRuntimeArgsBatch;
    SoftmaxCls postproc;
    return postproc.batchProcess(modelOutput, prepRuntimeArgsBatch, *params,
                                 output);
  }
  case GenericPostParams::AlgoType::FPR_CLS: {
    std::vector<FrameTransformContext> prepRuntimeArgsBatch;
    FprCls postproc;
    return postproc.batchProcess(modelOutput, prepRuntimeArgsBatch, *params,
                                 output);
  }
  case GenericPostParams::AlgoType::RAW_FEATURE: {
    std::vector<FrameTransformContext> prepRuntimeArgsBatch;
    RawFeature postproc;
    return postproc.batchProcess(modelOutput, prepRuntimeArgsBatch, *params,
                                 output);
  }
  case GenericPostParams::AlgoType::UNET_DUAL_OUTPUT: {
    if (!runtimeContext->has<std::vector<FrameTransformContext>>(
            "preproc_runtime_args_batch")) {
      LOG_ERRORS << "preproc_runtime_args_batch is nullptr";
      throw std::runtime_error("preproc_runtime_args_batch is nullptr");
    }
    const auto &prepRuntimeArgsBatch =
        runtimeContext->getParam<std::vector<FrameTransformContext>>(
            "preproc_runtime_args_batch");
    UNetDaulOutputSeg postproc;
    return postproc.batchProcess(modelOutput, prepRuntimeArgsBatch, *params,
                                 output);
  }
  case GenericPostParams::AlgoType::OCR_RECO: {
    std::vector<FrameTransformContext> prepRuntimeArgsBatch;
    OCRReco postproc;
    return postproc.batchProcess(modelOutput, prepRuntimeArgsBatch, *params,
                                 output);
  }
  default: {
    LOG_ERRORS << "Unknown generic algorithm type: "
               << static_cast<int>(params->algoType);
    return false;
  }
  }
  return true;
}
} // namespace ai_core::dnn