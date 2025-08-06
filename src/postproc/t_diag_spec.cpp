/**
 * @file t_diag_spec.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-06
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "t_diag_spec.hpp"
#include "ai_core/algo_output_types.hpp"
#include <logger.hpp>
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool TDiagSpecPostproc::process(const TensorData &modelOutput,
                                const FramePreprocessArg &prepArgs,
                                AlgoOutput &algoOutput,
                                const GenericPostParams &postArgs) const {

  if (postArgs.outputNames.size() != 8) {
    LOG_ERRORS << "TDiagSpecPostproc expects exactly 8 output names.";
    return false;
  }
  const auto &structureOutputName = postArgs.outputNames.at(0);
  const auto &eccentricOutputName = postArgs.outputNames.at(1);
  const auto &marginOutputName = postArgs.outputNames.at(2);
  const auto &aspectRatioOutputName = postArgs.outputNames.at(3);
  const auto &echoOutputName = postArgs.outputNames.at(4);
  const auto &focalEchoOutputName = postArgs.outputNames.at(5);
  const auto &tiradsOutputName = postArgs.outputNames.at(6);
  const auto &featOutputName = postArgs.outputNames.at(7);

  const auto &outputs = modelOutput.datas;
  const auto &outputShapes = modelOutput.shapes;

  TDiagSpecRet ret;
  // tirads
  if (outputShapes.at(tiradsOutputName).back() != 1) {
    LOG_ERRORS << "tiradsOutputName output shape is not 1.";
    return false;
  }
  ret.tirads = outputs.at(tiradsOutputName).getHostPtr<float>()[0];

  // feat
  ret.feat.assign(outputs.at(featOutputName).getHostPtr<float>(),
                  outputs.at(featOutputName).getHostPtr<float>() +
                      outputs.at(featOutputName).getElementCount());

  // structure
  if (outputShapes.at(structureOutputName).back() != 5) {
    LOG_ERRORS << "structureOutputName output shape is not 5.";
    return false;
  }
  std::copy(outputs.at(structureOutputName).getHostPtr<float>(),
            outputs.at(structureOutputName).getHostPtr<float>() + 5,
            ret.structure.begin());

  // eccentric
  if (outputShapes.at(eccentricOutputName).back() != 2) {
    LOG_ERRORS << "eccentricOutputName output shape is not 2.";
    return false;
  }
  std::copy(outputs.at(eccentricOutputName).getHostPtr<float>(),
            outputs.at(eccentricOutputName).getHostPtr<float>() + 2,
            ret.eccentric.begin());

  // margin
  if (outputShapes.at(marginOutputName).back() != 3) {
    LOG_ERRORS << "marginOutputName output shape is not 3.";
    return false;
  }
  std::copy(outputs.at(marginOutputName).getHostPtr<float>(),
            outputs.at(marginOutputName).getHostPtr<float>() + 3,
            ret.margin.begin());

  // aspectRatio
  if (outputShapes.at(aspectRatioOutputName).back() != 2) {
    LOG_ERRORS << "aspectRatioOutputName output shape is not 2.";
    return false;
  }
  std::copy(outputs.at(aspectRatioOutputName).getHostPtr<float>(),
            outputs.at(aspectRatioOutputName).getHostPtr<float>() + 2,
            ret.aspectRatio.begin());

  // echo
  if (outputShapes.at(echoOutputName).back() != 5) {
    LOG_ERRORS << "echoOutputName output shape is not 5.";
    return false;
  }
  std::copy(outputs.at(echoOutputName).getHostPtr<float>(),
            outputs.at(echoOutputName).getHostPtr<float>() + 5,
            ret.echo.begin());

  // focalEcho
  if (outputShapes.at(focalEchoOutputName).back() != 5) {
    LOG_ERRORS << "focalEchoOutputName output shape is not 5.";
    return false;
  }
  std::copy(outputs.at(focalEchoOutputName).getHostPtr<float>(),
            outputs.at(focalEchoOutputName).getHostPtr<float>() + 5,
            ret.focalEcho.begin());

  algoOutput.setParams(ret);

  return true;
}
} // namespace ai_core::dnn
