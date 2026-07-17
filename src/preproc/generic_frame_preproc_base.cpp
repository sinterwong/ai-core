/**
 * @file generic_frame_preproc_base.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "generic_frame_preproc_base.hpp"
#include "ai_core/logger.hpp"

namespace ai_core::dnn {

InferErrorCode GenericFramePreprocBase::process(
    const AlgoInput &input, const AlgoPreprocParams &params, TensorData &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  auto params_ptr = params.getParams<FramePreprocessArg>();
  if (params_ptr == nullptr) {
    LOG_ERROR_S << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return InferErrorCode::InferInvalidInput;
  }

  if (params_ptr->input_names.size() != 1) {
    LOG_ERROR_S << "Generic frame preprocessing expects exactly one input "
                   "name.";
    return InferErrorCode::InferInvalidInput;
  }

  auto frame_input = input.getParams<FrameInput>();
  if (!frame_input) {
    LOG_ERROR_S << "Unsupported AlgoInput type for generic frame "
                   "preprocessing.";
    return InferErrorCode::InferInvalidInput;
  }

  FrameTransformContext single_runtime_args;
  single_runtime_args.model_input_shape = params_ptr->model_input_shape;
  single_runtime_args.is_equal_scale = params_ptr->is_equal_scale;
  auto data = kernel().process(*params_ptr, *frame_input, single_runtime_args);
  runtime_context->setParam("preproc_runtime_args", single_runtime_args);
  output.datas.insert(std::make_pair(params_ptr->input_names[0], data));

  std::vector<int> shape;
  if (params_ptr->hwc2chw) {
    shape = {1, params_ptr->model_input_shape.c,
             params_ptr->model_input_shape.h, params_ptr->model_input_shape.w};
  } else {
    shape = {1, params_ptr->model_input_shape.h,
             params_ptr->model_input_shape.w, params_ptr->model_input_shape.c};
  }
  output.shapes.insert(std::make_pair(params_ptr->input_names[0], shape));
  return InferErrorCode::SUCCESS;
}

InferErrorCode GenericFramePreprocBase::batchProcess(
    const std::vector<AlgoInput> &inputs, const AlgoPreprocParams &params,
    TensorData &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {
  auto params_ptr = params.getParams<FramePreprocessArg>();
  if (params_ptr == nullptr) {
    LOG_ERROR_S << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return InferErrorCode::InferInvalidInput;
  }

  if (params_ptr->input_names.size() != 1) {
    LOG_ERROR_S << "Generic frame preprocessing expects exactly one input "
                   "name.";
    return InferErrorCode::InferInvalidInput;
  }

  std::vector<FrameInput> frame_inputs;
  frame_inputs.reserve(inputs.size());
  for (const auto &algo_input : inputs) {
    auto frame_input = algo_input.getParams<FrameInput>();
    if (!frame_input) {
      LOG_ERROR_S << "Unsupported AlgoInput type for generic frame "
                     "preprocessing.";
      return InferErrorCode::InferInvalidInput;
    }
    frame_inputs.push_back(*frame_input);
  }

  std::vector<FrameTransformContext> batch_runtime_args(inputs.size());
  auto data = kernel().batchProcess(*params_ptr, frame_inputs,
                                    batch_runtime_args);
  runtime_context->setParam("preproc_runtime_args_batch", batch_runtime_args);
  output.datas.insert(std::make_pair(params_ptr->input_names[0], data));

  std::vector<int> shape;
  if (params_ptr->hwc2chw) {
    shape = {static_cast<int>(inputs.size()), params_ptr->model_input_shape.c,
             params_ptr->model_input_shape.h, params_ptr->model_input_shape.w};
  } else {
    shape = {static_cast<int>(inputs.size()), params_ptr->model_input_shape.h,
             params_ptr->model_input_shape.w, params_ptr->model_input_shape.c};
  }
  output.shapes.insert(std::make_pair(params_ptr->input_names[0], shape));
  return InferErrorCode::SUCCESS;
}

} // namespace ai_core::dnn
