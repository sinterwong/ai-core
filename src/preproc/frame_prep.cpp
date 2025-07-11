/**
 * @file single_frame_prep.cpp
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
#include <ostream>

#ifdef WITH_NCNN
// #include "ncnn_generic_preprocessor.hpp"
#endif

#ifdef WITH_TRT
// #include "cuda_generic_preprocessor.hpp"
#endif

namespace ai_core::dnn {

bool FramePreprocess::process(AlgoInput &input, AlgoPreprocParams &params,
                              TensorData &output) {
  auto paramsPtr = params.getParams<FramePreprocessArg>();
  if (paramsPtr == nullptr) {
    LOG_ERRORS << "Failed to get FramePreprocessArg from AlgoPreprocParams.";
    return false;
  }
  auto frameInput = input.getParams<FrameInput>();
  if (!frameInput) {
    return false;
  }

  switch (paramsPtr->preprocTaskType) {
  case FramePreprocType::OPENCV_CPU_GENERIC: {
    std::unique_ptr<IFramePreprocessor> processor_ =
        std::make_unique<cpu::CpuGenericCvPreprocessor>();
    output.datas.insert(
        std::make_pair(paramsPtr->inputNames[0],
                       processor_->process(*paramsPtr, *frameInput)));
    std::vector<int> shape;
    if (paramsPtr->hwc2chw) {
      shape = {paramsPtr->modelInputShape.c, paramsPtr->modelInputShape.h,
               paramsPtr->modelInputShape.w};
    } else {
      shape = {paramsPtr->modelInputShape.h, paramsPtr->modelInputShape.w,
               paramsPtr->modelInputShape.c};
    }
    output.shapes.insert(std::make_pair(paramsPtr->inputNames[0], shape));
    break;
  }
  case FramePreprocType::NCNN_GENERIC: {
    LOG_ERRORS << "NCNN_GENERIC preprocessor requested, but WITH_NCNN is "
                  "not enabled.";
    return false;
    break;
  }
  case FramePreprocType::CUDA_GENERIC: {
    LOG_ERRORS << "CUDA_GENERIC preprocessor not implemented yet.";
    return false;
  }
  default: {
    LOG_ERRORS << "Unknown preprocessor type: "
               << static_cast<int>(paramsPtr->preprocTaskType);
    return false;
  }
  }
  return true;
}
} // namespace ai_core::dnn
