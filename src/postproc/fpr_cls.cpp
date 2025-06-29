/**
 * @file fpr_cls.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "fpr_cls.hpp"
#include "ai_core/types/algo_output_types.hpp"
#include "logger.hpp"

#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool FprCls::process(const TensorData &modelOutput, AlgoPreprocParams &prepArgs,
                     AlgoOutput &algoOutput, AlgoPostprocParams &postArgs) {
  if (modelOutput.datas.empty()) {
    LOG_ERRORS << "modelOutput.outputs is empty";
    return false;
  }

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  auto pScores = outputs.at("14");
  auto pBirads = outputs.at("15");

  std::vector<int> pScoresShape = outputShapes.at("14");
  int numClasses = pScoresShape.at(pScoresShape.size() - 1);

  std::vector<int> pBiradsShape = outputShapes.at("15");
  int numBirads = pBiradsShape.at(pBiradsShape.size() - 1);

  cv::Mat scores(1, numClasses, CV_32F,
                 const_cast<void *>(pScores.getTypedPtr<void>()));
  cv::Mat birads(1, numBirads, CV_32F,
                 const_cast<void *>(pBirads.getTypedPtr<void>()));

  cv::Point classIdPoint;
  double score;
  cv::minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

  cv::Point biradsIdPoint;
  double biradsScore;
  cv::minMaxLoc(birads, nullptr, &biradsScore, nullptr, &biradsIdPoint);

  FprClsRet fprRet;
  fprRet.score = score;
  fprRet.label = classIdPoint.x;
  fprRet.scoreProbs.assign(pScores.getTypedPtr<float>(),
                           pScores.getTypedPtr<float>() +
                               pScores.getElementCount());
  fprRet.birad = biradsIdPoint.x;
  algoOutput.setParams(fprRet);
  return true;
}
} // namespace ai_core::dnn
