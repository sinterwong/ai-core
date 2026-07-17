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
#include "ai_core/logger.hpp"
#include "postproc/fpr_cls.hpp"
#include "postproc/ocr_reco.hpp"
#include "postproc/raw_output.hpp"
#include "postproc/softmax_cls.hpp"
#include "postproc/unet_dual_out_seg.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
InferErrorCode CVGenericPostproc::process(
    const TensorData &model_output, const AlgoPostprocParams &post_args,
    AlgoOutput &algo_output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {

  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.outputs is empty";
    return InferErrorCode::InferOutputError;
  }

  auto params = post_args.getParams<GenericPostParams>();
  if (params == nullptr) {
    LOG_ERROR_S << "GenericPostParams params is nullptr";
    throw std::runtime_error("GenericPostParams params is nullptr");
  }

  switch (params->algo_type) {
  case GenericPostParams::AlgoType::SoftmaxCls: {
    FrameTransformContext prep_runtime_args;
    SoftmaxCls postproc;
    return postproc.process(model_output, prep_runtime_args, *params,
                            algo_output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  case GenericPostParams::AlgoType::FprCls: {
    FrameTransformContext prep_runtime_args;
    FprCls postproc;
    return postproc.process(model_output, prep_runtime_args, *params,
                            algo_output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  case GenericPostParams::AlgoType::RawModelOutput: {
    FrameTransformContext prep_runtime_args;
    RawModelOutput postproc;
    return postproc.process(model_output, prep_runtime_args, *params,
                            algo_output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  case GenericPostParams::AlgoType::UnetDualOutput: {
    if (!runtime_context->has<FrameTransformContext>("preproc_runtime_args")) {
      LOG_ERROR_S << "FramePreprocessArg is nullptr";
      throw std::runtime_error("FramePreprocessArg is nullptr");
    }
    const auto &prep_runtime_args =
        runtime_context->getParam<FrameTransformContext>(
            "preproc_runtime_args");
    UNetDualOutputSeg postproc;
    return postproc.process(model_output, prep_runtime_args, *params,
                            algo_output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  case GenericPostParams::AlgoType::OcrReco: {
    FrameTransformContext prep_runtime_args;
    OCRReco postproc;
    return postproc.process(model_output, prep_runtime_args, *params,
                            algo_output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  default: {
    LOG_ERROR_S << "Unknown generic algorithm type: "
                << static_cast<int>(params->algo_type);
    return InferErrorCode::InferOutputError;
  }
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode CVGenericPostproc::batchProcess(
    const TensorData &model_output, const AlgoPostprocParams &post_args,
    std::vector<AlgoOutput> &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.outputs is empty";
    return InferErrorCode::InferOutputError;
  }

  auto params = post_args.getParams<GenericPostParams>();
  if (params == nullptr) {
    LOG_ERROR_S << "GenericPostParams params is nullptr";
    throw std::runtime_error("GenericPostParams params is nullptr");
  }

  switch (params->algo_type) {
  case GenericPostParams::AlgoType::SoftmaxCls: {
    std::vector<FrameTransformContext> prep_runtime_args_batch;
    SoftmaxCls postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  case GenericPostParams::AlgoType::FprCls: {
    std::vector<FrameTransformContext> prep_runtime_args_batch;
    FprCls postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  case GenericPostParams::AlgoType::RawModelOutput: {
    std::vector<FrameTransformContext> prep_runtime_args_batch;
    RawModelOutput postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  case GenericPostParams::AlgoType::UnetDualOutput: {
    if (!runtime_context->has<std::vector<FrameTransformContext>>(
            "preproc_runtime_args_batch")) {
      LOG_ERROR_S << "preproc_runtime_args_batch is nullptr";
      throw std::runtime_error("preproc_runtime_args_batch is nullptr");
    }
    const auto &prep_runtime_args_batch =
        runtime_context->getParam<std::vector<FrameTransformContext>>(
            "preproc_runtime_args_batch");
    UNetDualOutputSeg postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  case GenericPostParams::AlgoType::OcrReco: {
    std::vector<FrameTransformContext> prep_runtime_args_batch;
    OCRReco postproc;
    return postproc.batchProcess(model_output, prep_runtime_args_batch, *params,
                                 output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }
  default: {
    LOG_ERROR_S << "Unknown generic algorithm type: "
                << static_cast<int>(params->algo_type);
    return InferErrorCode::InferOutputError;
  }
  }
  return InferErrorCode::SUCCESS;
}
} // namespace ai_core::dnn