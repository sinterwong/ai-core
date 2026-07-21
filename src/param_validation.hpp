/**
 * @file param_validation.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Structural validation of pre/postprocess parameters, run once when
 * they are bound at initialize().
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_PARAM_VALIDATION_HPP
#define AI_CORE_PARAM_VALIDATION_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/logger.hpp"

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
    if (arg->mean_vals.size() != arg->norm_vals.size()) {
      LOG_ERROR_S << "FramePreprocessArg: mean_vals and norm_vals must have "
                     "the same size.";
      return InferErrorCode::InferInvalidInput;
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
