/**
 * @file confidence_filter_postproc.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "confidence_filter_postproc.hpp"
#include "ai_core/postproc_types.hpp"
#include "semantic_seg.hpp"
#include <logger.hpp>
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool ConfidenceFilterPostproc::process(
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

  auto params = postArgs.getParams<ConfidenceFilterParams>();
  if (params == nullptr) {
    LOG_ERRORS << "GenericPostParams params is nullptr";
    throw std::runtime_error("GenericPostParams params is nullptr");
  }

  switch (params->algoType) {
  case ConfidenceFilterParams::AlgoType::SEMANTIC_SEG: {
    SemanticSeg postproc;
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
} // namespace ai_core::dnn