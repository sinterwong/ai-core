/**
 * @file anchor_det_postproc.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "anchor_det_postproc.hpp"
#include "postproc/nano_det.hpp"
#include "postproc/rtm_det.hpp"
#include "postproc/yolo_det.hpp"
#include <logger.hpp>
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool AnchorDetPostproc::process(
    const TensorData &modelOutput, const AlgoPostprocParams &postArgs,
    AlgoOutput &algoOutput,
    std::shared_ptr<RuntimeContext> &runtimeContext) const {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  if (!runtimeContext->has<FrameTransformContext>("preproc_runtime_args")) {
    LOG_ERRORS << "FramePreprocessArg is nullptr";
    throw std::runtime_error("FramePreprocessArg is nullptr");
  }

  const auto &prepRuntimeArgs =
      runtimeContext->getParam<FrameTransformContext>("preproc_runtime_args");

  auto params = postArgs.getParams<AnchorDetParams>();
  if (params == nullptr) {
    LOG_ERRORS << "AnchorDetParams params is nullptr";
    throw std::runtime_error("AnchorDetParams params is nullptr");
  }

  switch (params->algoType) {
  case AnchorDetParams::AlgoType::YOLO_DET_V11: {
    Yolov11Det postproc;
    return postproc.process(modelOutput, prepRuntimeArgs, *params, algoOutput);
  }
  case AnchorDetParams::AlgoType::RTM_DET: {
    RTMDet postproc;
    return postproc.process(modelOutput, prepRuntimeArgs, *params, algoOutput);
  }
  case AnchorDetParams::AlgoType::NANO_DET: {
    NanoDet postproc;
    return postproc.process(modelOutput, prepRuntimeArgs, *params, algoOutput);
  }
  default: {
    LOG_ERRORS << "Unknown detection algorithm type: "
               << static_cast<int>(params->algoType);
    return false;
  }
  }

  return true;
}

bool AnchorDetPostproc::batchProcess(
    const TensorData &modelOutput, const AlgoPostprocParams &postArgs,
    std::vector<AlgoOutput> &output,
    std::shared_ptr<RuntimeContext> &runtimeContext) const {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  if (!runtimeContext->has<std::vector<FrameTransformContext>>(
          "preproc_runtime_args_batch")) {
    LOG_ERRORS << "preproc_runtime_args_batch is nullptr";
    throw std::runtime_error("preproc_runtime_args_batch is nullptr");
  }

  const auto &prepRuntimeArgsBatch =
      runtimeContext->getParam<std::vector<FrameTransformContext>>(
          "preproc_runtime_args_batch");

  auto params = postArgs.getParams<AnchorDetParams>();
  if (params == nullptr) {
    LOG_ERRORS << "AnchorDetParams params is nullptr";
    throw std::runtime_error("AnchorDetParams params is nullptr");
  }

  switch (params->algoType) {
  case AnchorDetParams::AlgoType::YOLO_DET_V11: {
    Yolov11Det postproc;
    return postproc.batchProcess(modelOutput, prepRuntimeArgsBatch, *params,
                                 output);
  }
  case AnchorDetParams::AlgoType::RTM_DET: {
    RTMDet postproc;
    return postproc.batchProcess(modelOutput, prepRuntimeArgsBatch, *params,
                                 output);
  }
  case AnchorDetParams::AlgoType::NANO_DET: {
    NanoDet postproc;
    return postproc.batchProcess(modelOutput, prepRuntimeArgsBatch, *params,
                                 output);
  }
  default: {
    LOG_ERRORS << "Unknown detection algorithm type: "
               << static_cast<int>(params->algoType);
    return false;
  }
  }

  return true;
}
} // namespace ai_core::dnn