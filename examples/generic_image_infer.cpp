#include "generic_image_infer.hpp"
#include "algo_config_parser.hpp"
#include <logger.hpp>

namespace ai_core::example {
GenericImageInfer::GenericImageInfer(const std::string &configPath) {
  mParams = utils::AlgoConfigParser(configPath).parse();

  mEngine = std::make_shared<dnn::AlgoInference>(mParams.modelTypes,
                                                 mParams.inferParams);

  if (mEngine->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "engine initialize failed";
    throw std::runtime_error("Detector engine initialize failed");
  }
}

AlgoOutput GenericImageInfer::operator()(const cv::Mat &image,
                                         const cv::Rect &roi) {
  if (image.empty()) {
    LOG_ERRORS << "Input image is empty";
    return {};
  }

  cv::Rect detRoi = roi;
  if (detRoi.empty()) {
    detRoi = cv::Rect(0, 0, image.cols, image.rows);
  }

  AlgoInput algoInput;
  FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(image);
  frameInput.inputRoi = std::make_shared<cv::Rect>(detRoi);
  algoInput.setParams(frameInput);

  AlgoOutput algoOutput;
  if (mEngine->infer(algoInput, mParams.preprocParams, mParams.postprocParams,
                     algoOutput) != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "engine infer failed";
    return {};
  }

  return algoOutput;
}
} // namespace ai_core::example
