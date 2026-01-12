/**
 * @file frame_prep.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "frame_prep.hpp"
#include "ai_core/algo_data_types.hpp"
#include "ai_core/logger.hpp"
#include "cpu_generic_preprocessor.hpp"
#include "frame_preprocessor_base.hpp"
#include <opencv2/opencv.hpp>
#include <ostream>

#ifdef WITH_NCNN
// #include "ncnn_generic_preprocessor.hpp"
#endif

#ifdef WITH_TRT
#include "gpu_generic_cuda_preprocessor.hpp"
#endif

namespace ai_core::dnn {

bool FramePreprocess::process(
    const AlgoInput &input, const AlgoPreprocParams &params, TensorData &output,
    std::shared_ptr<RuntimeContext> &runtime_context) const {

  auto params_ptr = params.getParams<FramePreprocessArg>();
  if (params_ptr == nullptr) {
    LOG_ERROR_S << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return false;
  }

  if (params_ptr->input_names.size() != 1) {
    LOG_ERROR_S << "FramePreprocess expects exactly one input name.";
    return false;
  }

  auto frame_input = input.getParams<FrameInput>();

  if (!frame_input) {
    LOG_ERROR_S << "Unsupported AlgoInput type for FramePreprocess.";
    return false;
  }
  FrameTransformContext single_runtime_args;
  auto data = singleProcess(*params_ptr, *frame_input, single_runtime_args);
  runtime_context->setParam("preproc_runtime_args", single_runtime_args);
  output.datas.insert(std::make_pair(params_ptr->input_names[0], data));

  std::vector<int> shape;
  if (params_ptr->hwc2chw) {
    shape = {1, params_ptr->model_input_shape.c, params_ptr->model_input_shape.h,
             params_ptr->model_input_shape.w};
  } else {
    shape = {1, params_ptr->model_input_shape.h, params_ptr->model_input_shape.w,
             params_ptr->model_input_shape.c};
  }
  output.shapes.insert(std::make_pair(params_ptr->input_names[0], shape));
  return true;
}

bool FramePreprocess::batchProcess(
    const std::vector<AlgoInput> &input, const AlgoPreprocParams &params,
    TensorData &output, std::shared_ptr<RuntimeContext> &runtime_context) const {
  auto params_ptr = params.getParams<FramePreprocessArg>();
  if (params_ptr == nullptr) {
    LOG_ERROR_S << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return false;
  }

  if (params_ptr->input_names.size() != 1) {
    LOG_ERROR_S << "FramePreprocess expects exactly one input name.";
    return false;
  }

  std::vector<FrameInput> frame_inputs;
  frame_inputs.reserve(input.size());
  for (const auto &algo_input : input) {
    auto frame_input = algo_input.getParams<FrameInput>();
    if (!frame_input) {
      LOG_ERROR_S << "Unsupported AlgoInput type for FramePreprocess.";
      return false;
    }
    frame_inputs.push_back(*frame_input);
  }

  std::vector<FrameTransformContext> batch_runtime_args(input.size());
  auto data = batchProcess(*params_ptr, frame_inputs, batch_runtime_args);
  runtime_context->setParam("preproc_runtime_args_batch", batch_runtime_args);
  output.datas.insert(std::make_pair(params_ptr->input_names[0], data));

  std::vector<int> shape;
  if (params_ptr->hwc2chw) {
    shape = {static_cast<int>(input.size()), params_ptr->model_input_shape.c,
             params_ptr->model_input_shape.h, params_ptr->model_input_shape.w};
  } else {
    shape = {static_cast<int>(input.size()), params_ptr->model_input_shape.h,
             params_ptr->model_input_shape.w, params_ptr->model_input_shape.c};
  }
  output.shapes.insert(std::make_pair(params_ptr->input_names[0], shape));
  return true;
}

TypedBuffer
FramePreprocess::singleProcess(const FramePreprocessArg &args,
                               const FrameInput &input,
                               FrameTransformContext &runtime_context) const {
  runtime_context.model_input_shape = args.model_input_shape;
  runtime_context.is_equal_scale = args.is_equal_scale;
  switch (args.preproc_task_type) {
  case FramePreprocessArg::FramePreprocType::OpencvCpuGeneric: {
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<cpu::CpuGenericCvPreprocessor>();
    return processor->process(args, input, runtime_context);
  }
  case FramePreprocessArg::FramePreprocType::NcnnGeneric: {
#ifdef WITH_NCNN
    LOG_ERROR_S << "NCNN_GENERIC preprocessor requested, but not implemented.";
    throw std::runtime_error("NCNN_GENERIC preprocessor not implemented.");
#else
    LOG_ERROR_S << "NCNN_GENERIC preprocessor requested, but WITH_NCNN is "
                   "not enabled.";
    throw std::runtime_error(
        "NCNN_GENERIC preprocessor requested, but WITH_NCNN is not enabled.");
#endif
  }
  case FramePreprocessArg::FramePreprocType::CudaGpuGeneric: {
#ifdef WITH_TRT
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<gpu::GpuGenericCudaPreprocessor>();
    return processor->process(args, input, runtime_context);
#else
    LOG_ERROR_S << "CUDA_GPU_GENERIC preprocessor requested, but WITH_TRT is "
                   "not enabled.";
    throw std::runtime_error("CUDA_GPU_GENERIC preprocessor requested, but "
                             "WITH_TRT is not enabled.");
#endif
  }
  default: {
    LOG_ERROR_S << "Unknown preprocessor type: "
                << static_cast<int>(args.preproc_task_type);
    throw std::runtime_error("Unknown preprocessor type.");
  }
  }
}

TypedBuffer FramePreprocess::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &input,
    std::vector<FrameTransformContext> &runtime_context) const {
  switch (args.preproc_task_type) {
  case FramePreprocessArg::FramePreprocType::OpencvCpuGeneric: {
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<cpu::CpuGenericCvPreprocessor>();
    return processor->batchProcess(args, input, runtime_context);
  }
  case FramePreprocessArg::FramePreprocType::NcnnGeneric: {
#ifdef WITH_NCNN
    LOG_ERROR_S << "NCNN_GENERIC preprocessor requested, but not "
                   "implemented.";
    throw std::runtime_error(
        "BATCH_NCNN_GENERIC preprocessor not implemented.");
#else
    LOG_ERROR_S
        << "BATCH_NCNN_GENERIC preprocessor requested, but WITH_NCNN is "
           "not enabled.";
    throw std::runtime_error("BATCH_NCNN_GENERIC preprocessor requested, but "
                             "WITH_NCNN is not enabled.");
#endif
  }
  case FramePreprocessArg::FramePreprocType::CudaGpuGeneric: {
#ifdef WITH_TRT
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<gpu::GpuGenericCudaPreprocessor>();
    return processor->batchProcess(args, input, runtime_context);
#else
    LOG_ERROR_S
        << "BATCH_CUDA_GPU_GENERIC preprocessor requested, but WITH_TRT "
           "is not enabled.";
    throw std::runtime_error("BATCH_CUDA_GPU_GENERIC preprocessor requested, "
                             "but WITH_TRT is not enabled.");
#endif
  }
  default: {
    LOG_ERROR_S << "Unknown batch preprocessor type: "
                << static_cast<int>(args.preproc_task_type);
    throw std::runtime_error("Unknown batch preprocessor type.");
  }
  }
}
} // namespace ai_core::dnn
