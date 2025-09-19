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
#include "cpu_generic_preprocessor.hpp"
#include "frame_preprocessor_base.hpp"
#include <logger.hpp>
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
    std::shared_ptr<RuntimeContext> &runtimeContext) const {

  auto paramsPtr = params.getParams<FramePreprocessArg>();
  if (paramsPtr == nullptr) {
    LOG_ERRORS << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return false;
  }

  if (paramsPtr->inputNames.size() != 1) {
    LOG_ERRORS << "FramePreprocess expects exactly one input name.";
    return false;
  }

  auto frameInput = input.getParams<FrameInput>();

  if (!frameInput) {
    LOG_ERRORS << "Unsupported AlgoInput type for FramePreprocess.";
    return false;
  }
  FrameTransformContext singleRuntimeArgs;
  auto data = singleProcess(*paramsPtr, *frameInput, singleRuntimeArgs);
  runtimeContext->setParam("preproc_runtime_args", singleRuntimeArgs);
  output.datas.insert(std::make_pair(paramsPtr->inputNames[0], data));

  std::vector<int> shape;
  if (paramsPtr->hwc2chw) {
    shape = {1, paramsPtr->modelInputShape.c, paramsPtr->modelInputShape.h,
             paramsPtr->modelInputShape.w};
  } else {
    shape = {1, paramsPtr->modelInputShape.h, paramsPtr->modelInputShape.w,
             paramsPtr->modelInputShape.c};
  }
  output.shapes.insert(std::make_pair(paramsPtr->inputNames[0], shape));
  return true;
}

bool FramePreprocess::batchProcess(
    const std::vector<AlgoInput> &input, const AlgoPreprocParams &params,
    TensorData &output, std::shared_ptr<RuntimeContext> &runtimeContext) const {
  auto paramsPtr = params.getParams<FramePreprocessArg>();
  if (paramsPtr == nullptr) {
    LOG_ERRORS << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return false;
  }

  if (paramsPtr->inputNames.size() != 1) {
    LOG_ERRORS << "FramePreprocess expects exactly one input name.";
    return false;
  }

  std::vector<FrameInput> frameInputs;
  frameInputs.reserve(input.size());
  for (const auto &algoInput : input) {
    auto frameInput = algoInput.getParams<FrameInput>();
    if (!frameInput) {
      LOG_ERRORS << "Unsupported AlgoInput type for FramePreprocess.";
      return false;
    }
    frameInputs.push_back(*frameInput);
  }

  std::vector<FrameTransformContext> batchRuntimeArgs(input.size());
  auto data = batchProcess(*paramsPtr, frameInputs, batchRuntimeArgs);
  runtimeContext->setParam("preproc_runtime_args_batch", batchRuntimeArgs);
  output.datas.insert(std::make_pair(paramsPtr->inputNames[0], data));

  std::vector<int> shape;
  if (paramsPtr->hwc2chw) {
    shape = {static_cast<int>(input.size()), paramsPtr->modelInputShape.c,
             paramsPtr->modelInputShape.h, paramsPtr->modelInputShape.w};
  } else {
    shape = {static_cast<int>(input.size()), paramsPtr->modelInputShape.h,
             paramsPtr->modelInputShape.w, paramsPtr->modelInputShape.c};
  }
  output.shapes.insert(std::make_pair(paramsPtr->inputNames[0], shape));
  return true;
}

TypedBuffer
FramePreprocess::singleProcess(const FramePreprocessArg &args,
                               const FrameInput &input,
                               FrameTransformContext &runtimeContext) const {
  runtimeContext.modelInputShape = args.modelInputShape;
  runtimeContext.isEqualScale = args.isEqualScale;
  switch (args.preprocTaskType) {
  case FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC: {
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<cpu::CpuGenericCvPreprocessor>();
    return processor->process(args, input, runtimeContext);
  }
  case FramePreprocessArg::FramePreprocType::NCNN_GENERIC: {
#ifdef WITH_NCNN
    LOG_ERRORS << "NCNN_GENERIC preprocessor requested, but not implemented.";
    throw std::runtime_error("NCNN_GENERIC preprocessor not implemented.");
#else
    LOG_ERRORS << "NCNN_GENERIC preprocessor requested, but WITH_NCNN is "
                  "not enabled.";
    throw std::runtime_error(
        "NCNN_GENERIC preprocessor requested, but WITH_NCNN is not enabled.");
#endif
  }
  case FramePreprocessArg::FramePreprocType::CUDA_GPU_GENERIC: {
#ifdef WITH_TRT
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<gpu::GpuGenericCudaPreprocessor>();
    return processor->process(args, input, runtimeContext);
#else
    LOG_ERRORS << "CUDA_GPU_GENERIC preprocessor requested, but WITH_TRT is "
                  "not enabled.";
    throw std::runtime_error("CUDA_GPU_GENERIC preprocessor requested, but "
                             "WITH_TRT is not enabled.");
#endif
  }
  default: {
    LOG_ERRORS << "Unknown preprocessor type: "
               << static_cast<int>(args.preprocTaskType);
    throw std::runtime_error("Unknown preprocessor type.");
  }
  }
}

TypedBuffer FramePreprocess::batchProcess(
    const FramePreprocessArg &args, const std::vector<FrameInput> &input,
    std::vector<FrameTransformContext> &runtimeContext) const {
  switch (args.preprocTaskType) {
  case FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC: {
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<cpu::CpuGenericCvPreprocessor>();
    return processor->batchProcess(args, input, runtimeContext);
  }
  case FramePreprocessArg::FramePreprocType::NCNN_GENERIC: {
#ifdef WITH_NCNN
    LOG_ERRORS << "NCNN_GENERIC preprocessor requested, but not "
                  "implemented.";
    throw std::runtime_error(
        "BATCH_NCNN_GENERIC preprocessor not implemented.");
#else
    LOG_ERRORS << "BATCH_NCNN_GENERIC preprocessor requested, but WITH_NCNN is "
                  "not enabled.";
    throw std::runtime_error("BATCH_NCNN_GENERIC preprocessor requested, but "
                             "WITH_NCNN is not enabled.");
#endif
  }
  case FramePreprocessArg::FramePreprocType::CUDA_GPU_GENERIC: {
#ifdef WITH_TRT
    std::unique_ptr<IFramePreprocessor> processor =
        std::make_unique<gpu::GpuGenericCudaPreprocessor>();
    return processor->batchProcess(args, input, runtimeContext);
#else
    LOG_ERRORS << "BATCH_CUDA_GPU_GENERIC preprocessor requested, but WITH_TRT "
                  "is not enabled.";
    throw std::runtime_error("BATCH_CUDA_GPU_GENERIC preprocessor requested, "
                             "but WITH_TRT is not enabled.");
#endif
  }
  default: {
    LOG_ERRORS << "Unknown batch preprocessor type: "
               << static_cast<int>(args.preprocTaskType);
    throw std::runtime_error("Unknown batch preprocessor type.");
  }
  }
}
} // namespace ai_core::dnn
