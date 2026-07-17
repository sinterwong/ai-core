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
#include "ai_core/logger.hpp"
#include <opencv2/opencv.hpp>

namespace ai_core::dnn {
bool RawModelOutput::processTyped(const TensorData &model_output,
                             const FrameTransformContext &prep_args,
                             const GenericPostParams &post_args,
                             AlgoOutput &algo_output) const {

  if (model_output.datas.empty()) {
    LOG_ERROR_S << "model_output.outputs is empty";
    return false;
  }
  algo_output.setParams(model_output);
  return true;
}

bool RawModelOutput::batchProcessTyped(
    const TensorData &model_output,
    const std::vector<FrameTransformContext> &prep_args,
    const GenericPostParams &post_args,
    std::vector<AlgoOutput> &algo_output) const {

  LOG_ERROR_S << "RawModelOutput does not support batch processing right not!";
  return false;
}

} // namespace ai_core::dnn
