#pragma once

#include <ai_core/algo_infer.hpp>
#include <opencv2/opencv.hpp>
#include <string>

namespace ai_core::example {

class GenericImageInfer {
public:
  GenericImageInfer(const std::string &configPath);
  ~GenericImageInfer() = default;

  ai_core::AlgoOutput operator()(const cv::Mat &image, const cv::Rect &roi);

private:
  ai_core::AlgoPreprocParams
  extractPreprocParams(const ai_core::AlgoConstructParams &params);

  ai_core::AlgoPostprocParams
  extractPostprocParams(const ai_core::AlgoConstructParams &params);

private:
  std::shared_ptr<ai_core::dnn::AlgoInference> mEngine;
  ai_core::AlgoConstructParams mParams;
};

} // namespace ai_core::example
