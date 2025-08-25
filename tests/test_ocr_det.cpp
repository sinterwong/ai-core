/**
 * @file test_ocr_det.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/algo_input_types.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "infer_base.hpp"
#include "postproc/confidence_filter_postproc.hpp"
#include "postproc_base.hpp"
#include "preproc/frame_prep.hpp"
#include "preproc_base.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <functional>
#include <opencv2/imgcodecs.hpp>
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

namespace testing_ocr_det {
namespace fs = std::filesystem;

using namespace ai_core;
using namespace ai_core::dnn;

struct TestConfig {
  std::string testName;

  std::function<std::shared_ptr<InferBase>(const AlgoConstructParams &)>
      engineFactory;

  std::string modelPath;
  DataType inferDataType;
  DataType preprocDataType;
  DeviceType deviceType;
  std::string inputName;
  FramePreprocessArg::FramePreprocType preprocTaskType =
      FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC;
  BufferLocation bufferLocation = BufferLocation::CPU;
  bool needDecrypt = false;
  std::string decryptkeyStr = "";
};

class OCRDetInferenceTest : public ::testing::TestWithParam<TestConfig> {
protected:
  void SetUp() override {
    framePreproc = std::make_shared<FramePreprocess>();
    ASSERT_NE(framePreproc, nullptr);

    confidenceFilterPostproc = std::make_shared<ConfidenceFilterPostproc>();
    ASSERT_NE(confidenceFilterPostproc, nullptr);
  }

  void CheckResults(const SegRet *segRet) {
    ASSERT_NE(segRet, nullptr);
    ASSERT_EQ(segRet->clsToContours.size(), 1);
    ASSERT_EQ(segRet->clsToContours.at(1).size(), 28);
  }

  fs::path resourceDir = fs::path("assets");
  fs::path confDir = resourceDir / "conf";
  fs::path dataDir = resourceDir / "data";

  std::string imagePath = (dataDir / "ocr_det/image.png").string();

  std::shared_ptr<PreprocssBase> framePreproc;
  std::shared_ptr<PostprocssBase> confidenceFilterPostproc;
};

TEST_P(OCRDetInferenceTest, Normal) {
  const auto &config = GetParam();

  AlgoConstructParams tempInferParams;
  AlgoInferParams inferParams;
  inferParams.dataType = config.inferDataType;
  inferParams.modelPath = config.modelPath;
  inferParams.name = "ocr_det";
  inferParams.deviceType = config.deviceType;
  inferParams.needDecrypt = config.needDecrypt;
  inferParams.decryptkeyStr = config.decryptkeyStr;
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine = config.engineFactory(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  engine->prettyPrintModelInfos();

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {512, 512, 3};
  framePreprocessArg.dataType = config.preprocDataType;
  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {123.675f, 116.28f, 103.53f};
  framePreprocessArg.normVals = {58.395f, 57.12f, 57.375f};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = {config.inputName};
  framePreprocessArg.preprocTaskType = config.preprocTaskType;
  framePreprocessArg.outputLocation = config.bufferLocation;
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  ConfidenceFilterParams confidenceFilterParams;
  confidenceFilterParams.algoType =
      ConfidenceFilterParams::AlgoType::SEMANTIC_SEG;
  confidenceFilterParams.condThre = 0.3f;
  confidenceFilterParams.outputNames = {"sigmoid_0.tmp_0"};
  postprocParams.setParams(confidenceFilterParams);

  AlgoInput algoInput;
  FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(image);
  frameInput.inputRoi =
      std::make_shared<cv::Rect>(0, 0, image.cols, image.rows);
  algoInput.setParams(frameInput);

  TensorData modelInput;
  framePreproc->process(algoInput, preprocParams, modelInput);

  TensorData modelOutput;
  ASSERT_EQ(engine->infer(modelInput, modelOutput), InferErrorCode::SUCCESS);

  AlgoOutput algoOutput;
  ASSERT_TRUE(confidenceFilterPostproc->process(modelOutput, preprocParams,
                                                algoOutput, postprocParams));

  auto *segRet = algoOutput.getParams<SegRet>();
  CheckResults(segRet);

  cv::Mat visImage = image.clone();
  for (const auto &pair : segRet->clsToContours) {
    for (const auto &contour : pair.second) {
      cv::drawContours(visImage, std::vector<std::vector<cv::Point>>{contour},
                       -1, cv::Scalar(0, 255, 0), 2);
    }
  }
  std::string output_filename = "vis_ocr_det_" + config.testName + ".png";
  cv::imwrite(output_filename, visImage);
}

std::vector<TestConfig> GetTestConfigs() {
  std::vector<TestConfig> configs;
#ifdef WITH_ORT
  configs.push_back({"ort",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "assets/models/ch_PP_ocr_det.onnx", DataType::FLOAT32,
                     DataType::FLOAT32, DeviceType::CPU, "x",
                     FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
                     BufferLocation::CPU, false});
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

INSTANTIATE_TEST_SUITE_P(OCRDetInferenceBackends, OCRDetInferenceTest,
                         ::testing::ValuesIn(GetTestConfigs()),
                         testNameGenerator);

} // namespace testing_ocr_det