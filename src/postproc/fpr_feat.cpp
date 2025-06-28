/**
 * @file fpr_feat.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "fpr_feat.hpp"
#include "logger.hpp"

namespace ai_core::dnn {
bool FprFeature::process(const TensorData &modelOutput,
                         AlgoPreprocParams &prepArgs, AlgoOutput &algoOutput,
                         AlgoPostprocParams &postArgs) {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  // just one output
  auto output = outputs.at("171");
  auto outputShape = outputShapes.at("171");

  FeatureRet ret;
  ret.feature.assign(output.getTypedPtr<float>(),
                     output.getTypedPtr<float>() + output.getElementCount());
  ret.featSize = outputShape.at(outputShape.size() - 1);
  algoOutput.setParams(ret);
  return true;
}
} // namespace ai_core::dnn
