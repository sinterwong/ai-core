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
#include "logger.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool SoftmaxCls::process(const TensorData &modelOutput,
                         AlgoPreprocParams &prepArgs, AlgoOutput &algoOutput,
                         AlgoPostprocParams &postArgs) {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  auto postParams = postArgs.getParams<GenericPostParams>();
  if (postParams == nullptr) {
    LOG_ERRORS << "SoftmaxCls::process: GenericPostParams is nullptr";
    throw std::runtime_error(
        "SoftmaxCls::process: GenericPostParams is nullptr");
  }

  const auto &scoreOutputName = postParams->outputNames.at(0);

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  // just one output
  if (outputs.size() != 1) {
    LOG_ERRORS << "SoftmaxCls unexpected size of outputs " << outputs.size();
    throw std::runtime_error("SoftmaxCls  unexpected size of outputs");
  }
  auto output = outputs.at(scoreOutputName);

  std::vector<int> outputShape = outputShapes.at(scoreOutputName);
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
} // namespace ai_core::dnn
