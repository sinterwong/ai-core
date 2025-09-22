#pragma once

#include "algo_config_parser.hpp"
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
  std::shared_ptr<ai_core::dnn::AlgoInference> mEngine;
  utils::AlgoConfigData mParams;
};

} // namespace ai_core::example
