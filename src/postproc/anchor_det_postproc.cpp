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
#include "ai_core/logger.hpp"
#include "postproc/nano_det.hpp"
#include "postproc/rtm_det.hpp"
#include "postproc/yolo_det.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool AnchorDetPostproc::process(
    const TensorData &model_output, const AlgoPostprocParams &post_args,
    AlgoOutput &algo_output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.outputs is empty";
    return false;
  }

  if (!runtime_context->has<FrameTransformContext>("preproc_runtime_args")) {
    LOG_ERROR_S << "FramePreprocessArg is nullptr";
    throw std::runtime_error("FramePreprocessArg is nullptr");
  }

  const auto &prep_runtime_args =
      runtime_context->getParam<FrameTransformContext>("preproc_runtime_args");

  auto params = post_args.getParams<AnchorDetParams>();
  if (params == nullptr) {
    LOG_ERROR_S << "AnchorDetParams params is nullptr";
    throw std::runtime_error("AnchorDetParams params is nullptr");
  }

  switch (params->algo_type) {
  case AnchorDetParams::AlgoType::YoloDetV11: {
    Yolov11Det postproc;
    return postproc.process(model_output, prep_runtime_args, *params,
                            algo_output);
  }
  case AnchorDetParams::AlgoType::RtmDet: {
    RTMDet postproc;
    return postproc.process(model_output, prep_runtime_args, *params,
                            algo_output);
  }
  case AnchorDetParams::AlgoType::NanoDet: {
    NanoDet postproc;
    return postproc.process(model_output, prep_runtime_args, *params,
                            algo_output);
  }
  default: {
    LOG_ERROR_S << "Unknown detection algorithm type: "
                << static_cast<int>(params->algo_type);
    return false;
  }
  }

  return true;
}

bool AnchorDetPostproc::batchProcess(
    const TensorData &model_output, const AlgoPostprocParams &post_args,
    std::vector<AlgoOutput> &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.outputs is empty";
    return false;
  }

  if (!runtime_context->has<std::vector<FrameTransformContext>>(
          "preproc_runtime_args_batch")) {
    LOG_ERROR_S << "preproc_runtime_args_batch is nullptr";
    throw std::runtime_error("preproc_runtime_args_batch is nullptr");
  }

  const auto &prep_runtime_args_batch =
      runtime_context->getParam<std::vector<FrameTransformContext>>(
          "preproc_runtime_args_batch");

  auto params = post_args.getParams<AnchorDetParams>();
  if (params == nullptr) {
    LOG_ERROR_S << "AnchorDetParams params is nullptr";
    throw std::runtime_error("AnchorDetParams params is nullptr");
  }

  switch (params->algo_type) {
  case AnchorDetParams::AlgoType::YoloDetV11: {
    Yolov11Det postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output);
  }
  case AnchorDetParams::AlgoType::RtmDet: {
    RTMDet postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output);
  }
  case AnchorDetParams::AlgoType::NanoDet: {
    NanoDet postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output);
  }
  default: {
    LOG_ERROR_S << "Unknown detection algorithm type: "
                << static_cast<int>(params->algo_type);
    return false;
  }
  }

  return true;
}
} // namespace ai_core::dnn