/**
 * @file raw_output.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "raw_output.hpp"
#include <logger.hpp>
#include <opencv2/opencv.hpp>

namespace ai_core::dnn {
bool RawModelOutput::process(const TensorData &modelOutput,
                             const FrameTransformContext &prepArgs,
                             const GenericPostParams &postArgs,
                             AlgoOutput &algoOutput) const {

  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }
  algoOutput.setParams(modelOutput);
  return true;
}

bool RawModelOutput::batchProcess(
    const TensorData &modelOutput,
    const std::vector<FrameTransformContext> &prepArgs,
    const GenericPostParams &postArgs,
    std::vector<AlgoOutput> &algoOutput) const {

  LOG_ERRORS << "RawModelOutput does not support batch processing right not!";
  return false;
}

} // namespace ai_core::dnn
