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
#include "ai_core/logger.hpp"
#include "ai_core/postproc_types.hpp"
#include "semantic_seg.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool ConfidenceFilterPostproc::process(
    const TensorData &model_output, const AlgoPostprocParams &post_args,
    AlgoOutput &algo_output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.outputs is empty";
    return false;
  }

  auto params = post_args.getParams<ConfidenceFilterParams>();
  if (params == nullptr) {
    LOG_ERROR_S << "GenericPostParams params is nullptr";
    throw std::runtime_error("GenericPostParams params is nullptr");
  }

  switch (params->algo_type) {
  case ConfidenceFilterParams::AlgoType::SemanticSeg: {
    if (!runtime_context->has<FrameTransformContext>("preproc_runtime_args")) {
      LOG_ERROR_S << "FramePreprocessArg is nullptr";
      throw std::runtime_error("FramePreprocessArg is nullptr");
    }
    const auto &prep_runtime_args =
        runtime_context->getParam<FrameTransformContext>("preproc_runtime_args");

    SemanticSeg postproc;
    return postproc.process(model_output, prep_runtime_args, *params, algo_output);
  }
  default: {
    LOG_ERROR_S << "Unknown generic algorithm type: "
                << static_cast<int>(params->algo_type);
    return false;
  }
  }
  return true;
}

bool ConfidenceFilterPostproc::batchProcess(
    const TensorData &model_output, const AlgoPostprocParams &post_args,
    std::vector<AlgoOutput> &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.outputs is empty";
    return false;
  }

  auto params = post_args.getParams<ConfidenceFilterParams>();
  if (params == nullptr) {
    LOG_ERROR_S << "GenericPostParams params is nullptr";
    throw std::runtime_error("GenericPostParams params is nullptr");
  }

  switch (params->algo_type) {
  case ConfidenceFilterParams::AlgoType::SemanticSeg: {
    if (!runtime_context->has<std::vector<FrameTransformContext>>(
            "preproc_runtime_args_batch")) {
      LOG_ERROR_S << "FramePreprocessArg is nullptr";
      throw std::runtime_error("FramePreprocessArg is nullptr");
    }
    const auto &prep_runtime_args_batch =
        runtime_context->getParam<std::vector<FrameTransformContext>>(
            "preproc_runtime_args_batch");
    SemanticSeg postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output);
  }
  default: {
    LOG_ERROR_S << "Unknown generic algorithm type: "
                << static_cast<int>(params->algo_type);
    return false;
  }
  }
  return true;
}
} // namespace ai_core::dnn