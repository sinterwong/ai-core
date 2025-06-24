/**
 * @file softmax_cls.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "softmax_cls.hpp"
#include "infer_types.hpp"
#include "logger/logger.hpp"

namespace infer::dnn::vision {
bool SoftmaxCls::processOutput(const ModelOutput &modelOutput,
                               const FramePreprocessArg &args,
                               AlgoOutput &algoOutput) {
  if (modelOutput.outputs.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  const auto &outputShapes = modelOutput.outputShapes;
  const auto &outputs = modelOutput.outputs;

  // just one output
  if (outputs.size() != 1) {
    LOG_ERRORS << "SoftmaxCls unexpected size of outputs " << outputs.size();
    throw std::runtime_error("SoftmaxCls  unexpected size of outputs");
  }
  auto output = outputs.at("output");

  std::vector<int> outputShape = outputShapes.at("output");
  int numClasses = outputShape.at(outputShape.size() - 1);

  cv::Mat scores(1, numClasses, CV_32F,
                 const_cast<void *>(output.getTypedPtr<void>()));
  cv::Point classIdPoint;
  double score;
  cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

  ClsRet clsRet;
  clsRet.score = score;
  clsRet.label = classIdPoint.x;
  algoOutput.setParams(clsRet);
  return true;
}
} // namespace infer::dnn::vision
