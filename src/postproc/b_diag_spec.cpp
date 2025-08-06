/**
 * @file b_diag_spec.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "b_diag_spec.hpp"
#include "ai_core/algo_output_types.hpp"
#include <logger.hpp>
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool BDiagSpecPostproc::process(const TensorData &modelOutput,
                                const FramePreprocessArg &prepArgs,
                                AlgoOutput &algoOutput,
                                const GenericPostParams &postArgs) const {
  if (postArgs.outputNames.size() != 10) {
    LOG_ERRORS << "BDiagSpecPostproc expects exactly 9 output names.";
    return false;
  }
  const auto &maligScoreOutputName = postArgs.outputNames.at(0);
  const auto &irregularShapeOutputName = postArgs.outputNames.at(1);
  const auto &spiculationOutputName = postArgs.outputNames.at(2);
  const auto &blurOutputName = postArgs.outputNames.at(3);
  const auto &lesionClsOutputName = postArgs.outputNames.at(4);
  const auto &microlobulationOutputName = postArgs.outputNames.at(5);
  const auto &angularMarginsOutputName = postArgs.outputNames.at(6);
  const auto &deeperThanWideOutputName = postArgs.outputNames.at(7);
  const auto &calcificationOutputName = postArgs.outputNames.at(8);
  const auto &featOutputName = postArgs.outputNames.at(9);

  const auto &outputs = modelOutput.datas;
  const auto &outputShapes = modelOutput.shapes;

  BDiagSpecRet ret;
  // malignantScore
  ret.malignantScore = outputs.at(maligScoreOutputName).getHostPtr<float>()[0];

  // feat
  ret.feat.assign(outputs.at(featOutputName).getHostPtr<float>(),
                  outputs.at(featOutputName).getHostPtr<float>() +
                      outputs.at(featOutputName).getElementCount());

  // irregularShape
  ret.irregularShape =
      outputs.at(irregularShapeOutputName).getHostPtr<float>()[0];

  // spiculation
  ret.spiculation = outputs.at(spiculationOutputName).getHostPtr<float>()[0];

  // blur
  ret.blur = outputs.at(blurOutputName).getHostPtr<float>()[0];

  // lesionCls
  if (outputShapes.at(lesionClsOutputName).back() != 6) {
    LOG_ERRORS << "lesionClsOutputName output shape is not 6.";
    return false;
  }
  std::copy(outputs.at(lesionClsOutputName).getHostPtr<float>(),
            outputs.at(lesionClsOutputName).getHostPtr<float>() + 6,
            ret.lesionCls.begin());

  // microlobulation
  ret.microlobulation =
      outputs.at(microlobulationOutputName).getHostPtr<float>()[0];

  // angularMargins
  ret.angularMargins =
      outputs.at(angularMarginsOutputName).getHostPtr<float>()[0];

  // deeperThanwide
  ret.deeperThanwide =
      outputs.at(deeperThanWideOutputName).getHostPtr<float>()[0];

  // calcification
  ret.calcification =
      outputs.at(calcificationOutputName).getHostPtr<float>()[0];

  algoOutput.setParams(ret);

  return true;
}
} // namespace ai_core::dnn
