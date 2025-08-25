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
bool CVGenericPostproc::process(const TensorData &modelOutput,
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

  auto params = postArgs.getParams<GenericPostParams>();
  if (params == nullptr) {
    LOG_ERRORS << "GenericPostParams params is nullptr";
    throw std::runtime_error("GenericPostParams params is nullptr");
  }

  switch (params->algoType) {
  case GenericPostParams::AlogType::SOFTMAX_CLS: {
    SoftmaxCls postproc;
    return postproc.process(modelOutput, *prepParams, algoOutput, *params);
  }
  case GenericPostParams::AlogType::FPR_CLS: {
    FprCls postproc;
    return postproc.process(modelOutput, *prepParams, algoOutput, *params);
  }
  case GenericPostParams::AlogType::FPR_FEAT: {
    RawFeature postproc;
    return postproc.process(modelOutput, *prepParams, algoOutput, *params);
  }
  case GenericPostParams::AlogType::UNET_DUAL_OUTPUT: {
    UNetDaulOutputSeg postproc;
    return postproc.process(modelOutput, *prepParams, algoOutput, *params);
  }
  case GenericPostParams::AlogType::OCR_RECO: {
    OCRReco postproc;
    return postproc.process(modelOutput, *prepParams, algoOutput, *params);
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