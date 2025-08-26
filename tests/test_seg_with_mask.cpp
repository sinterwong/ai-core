#include "ai_core/algo_input_types.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/postproc_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "infer_base.hpp"
#include "postproc/cv_generic_postproc.hpp"
#include "postproc_base.hpp"
#include "preproc/frame_with_mask_prep.hpp"
#include "preproc_base.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <functional>
#include <logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#ifdef WITH_NCNN
#include "ncnn/dnn_infer.hpp"
#endif

#ifdef WITH_ORT
#include "ort/dnn_infer.hpp"
#endif

#ifdef WITH_TRT
#include "trt/dnn_infer.hpp"
#endif

namespace testing_seg_with_mask {
namespace fs = std::filesystem;

using namespace ai_core;
using namespace ai_core::dnn;

struct TestConfig {
  std::string testName;
  std::function<std::shared_ptr<InferBase>(const AlgoConstructParams &)>
      engineFactory;
  std::string modelFile;
  DataType inferDataType;
  DataType preprocDataType;
  DeviceType deviceType;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;

  bool needDecrypt = false;
  FramePreprocessArg::FramePreprocType preprocTaskType =
      FramePreprocessArg::FramePreprocType::OPENCV_CPU_CONCAT_MASK;
  BufferLocation bufferLocation = BufferLocation::CPU;
};

class SegWithMaskInferTest : public ::testing::TestWithParam<TestConfig> {
protected:
  void SetUp() override {
    Logger::LogConfig logConfig;
    logConfig.appName = "SegWithMask-Unit-Test";
    logConfig.logPath = "./logs";
    logConfig.logLevel = LogLevel::INFO;
    logConfig.enableConsole = true;
    logConfig.enableColor = true;
    Logger::instance()->initialize(logConfig);

    frameWithMaskPreproc = std::make_shared<FrameWithMaskPreprocess>();
    ASSERT_NE(frameWithMaskPreproc, nullptr);

    segPostproc = std::make_shared<CVGenericPostproc>();
    ASSERT_NE(segPostproc, nullptr);
  }

  fs::path resourceDir = fs::path("assets");
  fs::path confDir = resourceDir / "conf";
  fs::path dataDir = resourceDir / "data";
  fs::path modelDir = resourceDir / "models";

  std::string imagePath = (dataDir / "seg_with_mask/image.png").string();

  std::shared_ptr<PreprocssBase> frameWithMaskPreproc;
  std::shared_ptr<PostprocssBase> segPostproc;
};

TEST_P(SegWithMaskInferTest, Normal) {
  const auto &config = GetParam();

  AlgoConstructParams tempInferParams;
  AlgoInferParams inferParams;
  inferParams.dataType = config.inferDataType;
  inferParams.modelPath = (modelDir / config.modelFile).string();
  inferParams.name = "algo_seg_with_mask";
  inferParams.deviceType = config.deviceType;
  inferParams.needDecrypt = config.needDecrypt;
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine = config.engineFactory(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  engine->prettyPrintModelInfos();

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {320, 320, 2};
  framePreprocessArg.dataType = config.preprocDataType;
  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {120};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = config.inputNames;
  framePreprocessArg.preprocTaskType = config.preprocTaskType;
  framePreprocessArg.outputLocation = config.bufferLocation;
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  GenericPostParams genericParams;
  genericParams.algoType = GenericPostParams::AlgoType::UNET_DUAL_OUTPUT;
  genericParams.outputNames = config.outputNames;
  postprocParams.setParams(genericParams);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageGray;
  cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
  ASSERT_FALSE(image.empty());
  AlgoInput algoInput;
  FrameInputWithMask frameInputWithMask;
  frameInputWithMask.frameInput.image = std::make_shared<cv::Mat>(imageGray);
  frameInputWithMask.frameInput.inputRoi =
      std::make_shared<cv::Rect>(538, 84, 875, 659);
  frameInputWithMask.maskRegions = {cv::Rect(851, 416, 276, 76),
                                    cv::Rect(610, 398, 169, 48)};
  algoInput.setParams(frameInputWithMask);

  TensorData modelInput;
  frameWithMaskPreproc->process(algoInput, preprocParams, modelInput);

  TensorData modelOutput;
  ASSERT_EQ(engine->infer(modelInput, modelOutput), InferErrorCode::SUCCESS);

  AlgoOutput algoOutput;
  ASSERT_TRUE(segPostproc->process(modelOutput, preprocParams, algoOutput,
                                   postprocParams));

  auto *segRet = algoOutput.getParams<DaulRawSegRet>();
  ASSERT_NE(segRet, nullptr);
  ASSERT_NE(segRet->mask, nullptr);
  ASSERT_NE(segRet->prob, nullptr);

  cv::Mat visImage = image.clone();

  cv::Mat segMask;
  segRet->mask->convertTo(segMask, CV_8U);
  cv::Mat resizedMask;
  cv::resize(segMask, resizedMask,
             cv::Size(segRet->roi->width, segRet->roi->height), 0, 0,
             cv::INTER_NEAREST);
  cv::imwrite("vis_seg_raw_mask_class_indices.jpg", resizedMask * 60);

  std::vector<cv::Vec3b> colorPalette;
  colorPalette.push_back(cv::Vec3b(0, 0, 0));
  colorPalette.push_back(cv::Vec3b(0, 0, 255));
  colorPalette.push_back(cv::Vec3b(0, 255, 0));
  colorPalette.push_back(cv::Vec3b(255, 0, 0));
  colorPalette.push_back(cv::Vec3b(0, 255, 255));

  cv::Mat colorMask(resizedMask.size(), CV_8UC3);
  for (int i = 0; i < resizedMask.rows; ++i) {
    for (int j = 0; j < resizedMask.cols; ++j) {
      int class_idx = resizedMask.at<uchar>(i, j);
      if (class_idx >= 0 && class_idx < colorPalette.size()) {
        colorMask.at<cv::Vec3b>(i, j) = colorPalette[class_idx];
      } else {
        colorMask.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
      }
    }
  }
  cv::addWeighted(visImage(*segRet->roi), 0.8, colorMask, 0.2, 0,
                  visImage(*segRet->roi));

  std::string output_filename = "vis_seg_with_mask_" + config.testName + ".png";
  cv::imwrite(output_filename, visImage);
}

std::vector<TestConfig> GetTestConfigs() {
  std::vector<TestConfig> configs;
#ifdef WITH_ORT
  configs.push_back({"ort", // testName
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "seg_with_mask.onnx",
                     DataType::FLOAT32,
                     DataType::FLOAT32,
                     DeviceType::CPU,
                     {"argument_1.1"},
                     {"1368", "1369"}});
#endif
#ifdef WITH_NCNN
#endif
#ifdef WITH_TRT
#endif
  return configs;
}

std::string testNameGenerator(const testing::TestParamInfo<TestConfig> &info) {
  return info.param.testName;
}

INSTANTIATE_TEST_SUITE_P(SegWithMaskInferBackends, SegWithMaskInferTest,
                         ::testing::ValuesIn(GetTestConfigs()),
                         testNameGenerator);

} // namespace testing_seg_with_mask