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
bool AnchorDetPostproc::process(const TensorData &modelOutput,
                                AlgoPreprocParams &prepArgs,
                                AlgoOutput &algoOutput,
                                AlgoPostprocParams &postArgs) const {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  const auto &prepParams = prepArgs.getParams<FramePreprocessArg>();
  if (prepParams == nullptr) {
    LOG_ERRORS << "FramePreprocessArg is nullptr";
    throw std::runtime_error("FramePreprocessArg is nullptr");
  }

  auto params = postArgs.getParams<AnchorDetParams>();
  if (params == nullptr) {
    LOG_ERRORS << "AnchorDetParams params is nullptr";
    throw std::runtime_error("AnchorDetParams params is nullptr");
  }

  switch (params->algoType) {
  case AnchorDetParams::AlgoType::YOLO_DET_V11: {
    Yolov11Det postproc;
    return postproc.process(modelOutput, *prepParams, algoOutput, *params);
  }
  case AnchorDetParams::AlgoType::RTM_DET: {
    RTMDet postproc;
    return postproc.process(modelOutput, *prepParams, algoOutput, *params);
  }
  case AnchorDetParams::AlgoType::NANO_DET: {
    NanoDet postproc;
    return postproc.process(modelOutput, *prepParams, algoOutput, *params);
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