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
#include "ocr_reco.hpp"
#include "ai_core/algo_output_types.hpp"
#include <opencv2/core.hpp>

namespace ai_core::dnn {
bool OCRReco::process(const TensorData &modelOutput,
                      const FramePreprocessArg &prepArgs,
                      AlgoOutput &algoOutput,
                      const GenericPostParams &postArgs) const {
  const auto &outputLengthsName = postArgs.outputNames.at(0);
  const auto &argmaxOutputName = postArgs.outputNames.at(1);

  const auto &outputShapes = modelOutput.shapes;
  const auto &outputs = modelOutput.datas;

  OCRRecoRet ocrRet;
  auto pOutputLengths = outputs.at(outputLengthsName);
  auto pArgmaxOutputs = outputs.at(argmaxOutputName);

  ocrRet.outputLengths = *pOutputLengths.getHostPtr<int64_t>();

  std::vector<int64_t> argmaxOutputsVec(pArgmaxOutputs.getHostPtr<int64_t>(),
                                        pArgmaxOutputs.getHostPtr<int64_t>() +
                                            pArgmaxOutputs.getElementCount());
  ocrRet.outputs = ctcProcess(argmaxOutputsVec);

  algoOutput.setParams(ocrRet);
  return true;
}

std::vector<int64_t>
OCRReco::ctcProcess(const std::vector<int64_t> &outputs) const {
  std::vector<int64_t> result;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i] != 0 && !(i > 0 && outputs[i] == outputs[i - 1])) {
      result.push_back(outputs[i]);
    }
  }
  return result;
}
} // namespace ai_core::dnn
