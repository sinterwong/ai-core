#ifndef AI_CORE_PARAM_VALIDATION_HPP
#define AI_CORE_PARAM_VALIDATION_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "logger.hpp"
#include <algorithm>

namespace ai_core::dnn {

inline InferErrorCode validateBoundParams(const AlgoPreprocParams &params) {
  if (const auto *arg = params.getParams<FramePreprocessArg>()) {
    if (arg->model_input_shape.w <= 0 || arg->model_input_shape.h <= 0 ||
        arg->model_input_shape.c <= 0) {
      LOG_ERROR_S << "FramePreprocessArg: model_input_shape must be positive.";
      return InferErrorCode::InferInvalidInput;
    }
    if (arg->input_names.size() != 1) {
      LOG_ERROR_S << "FramePreprocessArg: exactly one input name required, "
                  << "got " << arg->input_names.size() << ".";
      return InferErrorCode::InferInvalidInput;
    }
    if (arg->mean_vals.size() != arg->std_vals.size()) {
      LOG_ERROR_S << "FramePreprocessArg: mean_vals and std_vals must have "
                     "the same size.";
      return InferErrorCode::InferInvalidInput;
    }
    // NB: model_input_format's channel count is deliberately not checked
    // against model_input_shape.c here — FrameWithMaskPreprocess legitimately
    // adds a mask channel on top of the colour channels. The kernels enforce
    // the invariant where it actually holds.

    // std_vals is a divisor: (v - mean) / std. Frameworks like NCNN and MNN
    // call the equivalent field a multiplier, so `1/255` gets written where
    // `255` was meant. That silently saturates every sigmoid downstream and
    // looks like a postprocessing bug, so say something.
    if (!arg->std_vals.empty() &&
        std::all_of(arg->std_vals.begin(), arg->std_vals.end(),
                    [](float v) { return v > 0.f && v < 1.f; })) {
      LOG_WARNING_S << "FramePreprocessArg: every std_vals entry is < 1 for an "
                       "8-bit input. std_vals is a DIVISOR — (v - mean) / std "
                       "— so scaling to [0,1] wants {255,...}, not {1/255,...}."
                       " Check whether it was written as a multiplier.";
    }
    if (arg->data_type != DataType::FLOAT32 &&
        arg->data_type != DataType::FLOAT16) {
      LOG_ERROR_S << "FramePreprocessArg: data_type must be FLOAT32 or "
                     "FLOAT16.";
      return InferErrorCode::InferInvalidInput;
    }
    return InferErrorCode::SUCCESS;
  }
  LOG_ERROR_S << "Preprocess parameters are empty (monostate); bind concrete "
                 "parameters at initialize().";
  return InferErrorCode::InferInvalidInput;
}

inline InferErrorCode validateBoundParams(const AlgoPostprocParams &params) {
  if (const auto *anchor = params.getParams<AnchorDetParams>()) {
    if (anchor->output_names.empty()) {
      LOG_ERROR_S << "AnchorDetParams: output_names must not be empty.";
      return InferErrorCode::InferInvalidInput;
    }
    return InferErrorCode::SUCCESS;
  }
  if (const auto *conf = params.getParams<ConfidenceFilterParams>()) {
    if (conf->output_names.empty()) {
      LOG_ERROR_S << "ConfidenceFilterParams: output_names must not be empty.";
      return InferErrorCode::InferInvalidInput;
    }
    return InferErrorCode::SUCCESS;
  }
  if (params.getParams<GenericPostParams>() != nullptr) {
    // output_names requirements are plugin-specific (RawModelOutput ignores
    // them entirely), so nothing structural to check here.
    return InferErrorCode::SUCCESS;
  }
  LOG_ERROR_S << "Postprocess parameters are empty (monostate); bind concrete "
                 "parameters at initialize().";
  return InferErrorCode::InferInvalidInput;
}

} // namespace ai_core::dnn

#endif // AI_CORE_PARAM_VALIDATION_HPP
