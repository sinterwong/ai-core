#pragma once

#include "algo_config_parser.hpp"
#include <ai_core/algo_infer_engine.hpp>
#include <ai_core/algo_postproc.hpp>
#include <ai_core/algo_preproc.hpp>
#include <vector>

namespace ai_core::example {
class OCRRec {

public:
  OCRRec(const std::string &modelPath, const std::string &dictPath = "");
  ~OCRRec();

  ai_core::OCRRecoRet process(const cv::Mat &imageGray);

  std::string mapToString(const std::vector<int64_t> &recResult);

private:
  utils::AlgoConfigData mParams;
  std::shared_ptr<ai_core::dnn::AlgoPreproc> mFramePreproc;
  std::shared_ptr<ai_core::dnn::AlgoInferEngine> mEngine;
  std::shared_ptr<ai_core::dnn::AlgoPostproc> mOcrPostproc;
  std::vector<wchar_t> mDict;

  int batch;
};

} // namespace ai_core::example