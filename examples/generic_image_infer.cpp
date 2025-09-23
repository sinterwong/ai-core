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

std::vector<ai_core::AlgoOutput>
GenericImageInfer::operator()(const std::vector<cv::Mat> &images,
                              const std::vector<cv::Rect> &rois) {
  if (images.empty()) {
    LOG_ERRORS << "Input images vector is empty";
    return {};
  }
  if (images.size() != rois.size()) {
    LOG_ERRORS << "Input images and rois vectors must have the same size";
    return {};
  }

  std::vector<AlgoInput> algoInputs(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    if (images[i].empty()) {
      LOG_WARNINGS << "Input image at index " << i << " is empty, skipping.";
      continue;
    }

    cv::Rect detRoi = rois[i];
    if (detRoi.empty()) {
      detRoi = cv::Rect(0, 0, images[i].cols, images[i].rows);
    }

    FrameInput frameInput;
    frameInput.image = std::make_shared<cv::Mat>(images[i]);
    frameInput.inputRoi = std::make_shared<cv::Rect>(detRoi);
    algoInputs[i].setParams(frameInput);
  }

  std::vector<AlgoOutput> algoOutputs;
  if (mEngine->batchInfer(algoInputs, mParams.preprocParams,
                          mParams.postprocParams,
                          algoOutputs) != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "engine batch infer failed";
    return {};
  }
  return algoOutputs;
}
} // namespace ai_core::example
