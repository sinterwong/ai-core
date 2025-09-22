#include "generic_image_infer.hpp"
#include "algo_config_parser.hpp"
#include <logger.hpp>
#include <nlohmann/json.hpp>

namespace ai_core::example {
GenericImageInfer::GenericImageInfer(const std::string &configPath) {
  mParams = utils::AlgoConfigParser(configPath).parse();

  AlgoModuleTypes moduleTypes;
  moduleTypes.preprocModule = mParams.getParam<std::string>("preprocType");
  moduleTypes.inferModule = mParams.getParam<std::string>("inferType");
  moduleTypes.postprocModule = mParams.getParam<std::string>("postprocType");
  AlgoInferParams inferParams =
      mParams.getParam<AlgoInferParams>("inferParams");
  std::string securityKey = SECURITY_KEY;
  inferParams.decryptkeyStr = securityKey;
  mEngine = std::make_shared<dnn::AlgoInference>(moduleTypes, inferParams);

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

  AlgoPreprocParams preprocParams = extractPreprocParams(mParams);
  AlgoPostprocParams postprocParams = extractPostprocParams(mParams);

  AlgoInput algoInput;
  FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(image);
  frameInput.inputRoi = std::make_shared<cv::Rect>(detRoi);
  algoInput.setParams(frameInput);

  AlgoOutput algoOutput;
  if (mEngine->infer(algoInput, preprocParams, postprocParams, algoOutput) !=
      InferErrorCode::SUCCESS) {
    LOG_ERRORS << "engine infer failed";
    return {};
  }

  return algoOutput;
}

AlgoPreprocParams
GenericImageInfer::extractPreprocParams(const AlgoConstructParams &params) {
  AlgoPreprocParams preprocParams;
  if (params.has<FramePreprocessArg>("preprocParams")) {
    const FramePreprocessArg &framePreprocessArg =
        params.getParam<FramePreprocessArg>("preprocParams");
    preprocParams.setParams(framePreprocessArg);
  } else {
    LOG_ERRORS << "Unsupported preprocParams type";
    throw std::runtime_error("Unsupported preprocParams type");
  }
  return preprocParams;
}

AlgoPostprocParams
GenericImageInfer::extractPostprocParams(const AlgoConstructParams &params) {
  AlgoPostprocParams postprocParams;
  if (params.has<AnchorDetParams>("postprocParams")) {
    const AnchorDetParams &anchorDetParams =
        params.getParam<AnchorDetParams>("postprocParams");
    postprocParams.setParams(anchorDetParams);
  } else if (params.has<GenericPostParams>("postprocParams")) {
    const GenericPostParams &genericPostParams =
        params.getParam<GenericPostParams>("postprocParams");
    postprocParams.setParams(genericPostParams);
  } else if (params.has<ConfidenceFilterParams>("postprocParams")) {
    const ConfidenceFilterParams &confidenceFilterParams =
        params.getParam<ConfidenceFilterParams>("postprocParams");
    postprocParams.setParams(confidenceFilterParams);
  } else {
    LOG_ERRORS << "Unsupported postprocParams type";
    throw std::runtime_error("Unsupported postprocParams type");
  }
  return postprocParams;
}

} // namespace ai_core::example
