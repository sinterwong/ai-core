#include "generic_image_infer.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/opencv_interop.hpp"
#include "algo_config_parser.hpp"

namespace ai_core::example {
GenericImageInfer::GenericImageInfer(const std::string &config_path) {
  mParams = utils::AlgoConfigParser(config_path).parse();

  mEngine = std::make_shared<dnn::AlgoInference>(mParams.modelTypes,
                                                 mParams.inferParams);

  if (mEngine->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "engine initialize failed";
    throw std::runtime_error("Detector engine initialize failed");
  }
}

AlgoOutput GenericImageInfer::operator()(const cv::Mat &image,
                                         const cv::Rect &roi) {
  if (image.empty()) {
    LOG_ERROR_S << "Input image is empty";
    return {};
  }

  cv::Rect det_roi = roi;
  if (det_roi.empty()) {
    det_roi = cv::Rect(0, 0, image.cols, image.rows);
  }

  AlgoInput algo_input;
  FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image);
  frame_input.roi = ai_core::interop::fromCv(det_roi);
  algo_input.setParams(frame_input);

  AlgoOutput algo_output;
  if (mEngine->infer(algo_input, mParams.preproc_params,
                     mParams.postproc_params,
                     algo_output) != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "engine infer failed";
    return {};
  }

  return algo_output;
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

  std::vector<AlgoInput> algo_inputs(images.size());
  for (size_t i = 0; i < images.size(); ++i) {
    if (images[i].empty()) {
      LOG_WARNINGS << "Input image at index " << i << " is empty, skipping.";
      continue;
    }

    cv::Rect det_roi = rois[i];
    if (det_roi.empty()) {
      det_roi = cv::Rect(0, 0, images[i].cols, images[i].rows);
    }

    FrameInput frame_input;
    frame_input.image = ai_core::interop::viewFromMat(images[i]);
    frame_input.roi = ai_core::interop::fromCv(det_roi);
    algo_inputs[i].setParams(frame_input);
  }

  std::vector<AlgoOutput> algo_outputs;
  if (mEngine->batchInfer(algo_inputs, mParams.preproc_params,
                          mParams.postproc_params,
                          algo_outputs) != InferErrorCode::SUCCESS) {
    LOG_ERRORS << "engine batch infer failed";
    return {};
  }
  return algo_outputs;
}
} // namespace ai_core::example
