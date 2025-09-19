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
                         const FrameTransformContext &prepArgs,
                         const GenericPostParams &postArgs,
                         AlgoOutput &algoOutput) const {

  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }
  const auto &featureOutputName = postArgs.outputNames.at(0);
  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  auto output = outputs.at(featureOutputName);

  std::vector<int> outputShape = outputShapes.at(featureOutputName);
  int numFeatures = outputShape.at(outputShape.size() - 1);

  FeatureRet ret = processSingleItem(output.getHostPtr<float>(), numFeatures);

  algoOutput.setParams(ret);
  return true;
}

bool RawFeature::batchProcess(
    const TensorData &modelOutput,
    const std::vector<FrameTransformContext> &prepArgs,
    const GenericPostParams &postArgs,
    std::vector<AlgoOutput> &algoOutput) const {
  const auto &featureOutputName = postArgs.outputNames.at(0);

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  auto output = outputs.at(featureOutputName);
  auto outputShape = outputShapes.at(featureOutputName);

  if (outputShape.size() < 2) {
    LOG_ERRORS << "RawFeature batchProcess expects output tensor with at least "
                  "2 dimensions (batchSize, numFeatures), but got "
               << outputShape.size();
    return false;
  }

  int batchSize = outputShape.at(0);
  int numFeatures = outputShape.at(outputShape.size() - 1);

  if (prepArgs.size() != batchSize) {
    LOG_ERRORS << "Batch size mismatch between model output (" << batchSize
               << ") and prepArgs (" << prepArgs.size() << ").";
    return false;
  }

  const float *allFeatureData = output.getHostPtr<float>();
  algoOutput.resize(batchSize);

  for (int i = 0; i < batchSize; ++i) {
    const float *currentFeatureData = allFeatureData + i * numFeatures;
    FeatureRet ret = processSingleItem(currentFeatureData, numFeatures);
    algoOutput[i].setParams(ret);
  }
  return true;
}

FeatureRet RawFeature::processSingleItem(const float *featureData,
                                         int numFeatures) const {
  FeatureRet ret;
  ret.feature.assign(featureData, featureData + numFeatures);
  ret.featSize = numFeatures;
  return ret;
}
} // namespace ai_core::dnn
