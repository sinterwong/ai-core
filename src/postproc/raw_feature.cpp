/**
 * @file raw_feature.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "raw_feature.hpp"
#include <logger.hpp>
#include <opencv2/opencv.hpp>

namespace ai_core::dnn {
bool RawFeature::process(const TensorData &modelOutput,
                         const FramePreprocessArg &prepArgs,
                         AlgoOutput &algoOutput,
                         const GenericPostParams &postArgs) const {
  const auto &featureOutputName = postArgs.outputNames.at(0);

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  // just one output
  auto output = outputs.at(featureOutputName);
  auto outputShape = outputShapes.at(featureOutputName);

  FeatureRet ret;
  ret.feature.assign(output.getHostPtr<float>(),
                     output.getHostPtr<float>() + output.getElementCount());
  ret.featSize = outputShape.at(outputShape.size() - 1);
  algoOutput.setParams(ret);
  return true;
}
} // namespace ai_core::dnn
