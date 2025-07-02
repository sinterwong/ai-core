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
#include <ostream>

#include "ai_core/algo_data_types.hpp"
#include "frame_prep.hpp"

#include <logger.hpp>

namespace ai_core::dnn {

FramePreprocess::FramePreprocess() {
  processor_ = std::make_unique<ImagePreprocessor>();
}

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

  output.datas.insert(std::make_pair(
      paramsPtr->inputName, processor_->process(*paramsPtr, *frameInput)));
  std::vector<int> shape;
  if (paramsPtr->hwc2chw) {
    shape = {paramsPtr->modelInputShape.c, paramsPtr->modelInputShape.h,
             paramsPtr->modelInputShape.w};
  } else {
    shape = {paramsPtr->modelInputShape.h, paramsPtr->modelInputShape.w,
             paramsPtr->modelInputShape.c};
  }
  output.shapes.insert(std::make_pair(paramsPtr->inputName, shape));
  return true;
}
} // namespace ai_core::dnn
