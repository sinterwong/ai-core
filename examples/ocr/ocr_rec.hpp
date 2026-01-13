#pragma once

#include "algo_config_parser.hpp"
#include "ai_core/infer_engine_wrapper.hpp"
#include "ai_core/algo_postprocessor.hpp"
#include "ai_core/algo_preprocessor.hpp"
#include <vector>

namespace ai_core::example {
class OCRRec {

public:
  OCRRec(const std::string &modelPath, const std::string &dictPath = "");
  ~OCRRec();

  ai_core::OCRRecoRet process(const cv::Mat &image_gray);

  std::string mapToString(const std::vector<int64_t> &recResult);

private:
  utils::AlgoConfigData mParams;
  std::shared_ptr<ai_core::dnn::AlgoPreproc> mFramePreproc;
  std::shared_ptr<ai_core::dnn::AlgoInferEngine> mEngine;
  std::shared_ptr<ai_core::dnn::AlgoPostproc> mOcrPostproc;
  std::vector<std::string> mDict;
};

} // namespace ai_core::example