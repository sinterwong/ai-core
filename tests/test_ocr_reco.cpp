/**
 * @file test_ocr_reco.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-25
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/algo_input_types.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "infer_base.hpp"
#include "postproc/cv_generic_postproc.hpp"
#include "postproc_base.hpp"
#include "preproc/frame_prep.hpp"
#include "preproc_base.hpp"
#include "gtest/gtest.h"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <logger.hpp>
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

namespace testing_ocr_reco {
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
  FramePreprocessArg::FramePreprocType preprocTaskType =
      FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC;
  BufferLocation bufferLocation = BufferLocation::CPU;
  bool needDecrypt = false;
  std::string decryptkeyStr = "";
};

class OCRRecoInferTest : public ::testing::TestWithParam<TestConfig> {
protected:
  void SetUp() override {
    Logger::LogConfig logConfig;
    logConfig.appName = "OCR-Unit-Test";
    logConfig.logPath = "./logs";
    logConfig.logLevel = LogLevel::INFO;
    logConfig.enableConsole = true;
    logConfig.enableColor = true;
    Logger::instance()->initialize(logConfig);

    framePreproc = std::make_shared<FramePreprocess>();
    ASSERT_NE(framePreproc, nullptr);

    ocrPostproc = std::make_shared<CVGenericPostproc>();
    ASSERT_NE(ocrPostproc, nullptr);
  }

  void CheckResults(const SegRet *segRet) {
    ASSERT_NE(segRet, nullptr);
    ASSERT_EQ(segRet->clsToContours.size(), 1);
    ASSERT_EQ(segRet->clsToContours.at(1).size(), 28);
  }

  fs::path resourceDir = fs::path("assets");
  fs::path confDir = resourceDir / "conf";
  fs::path dataDir = resourceDir / "data";

  std::string imagePath = (dataDir / "ocr_reco/image.png").string();

  std::shared_ptr<PreprocssBase> framePreproc;
  std::shared_ptr<PostprocssBase> ocrPostproc;
};

TEST_P(OCRRecoInferTest, Normal) {
  const auto &config = GetParam();

  AlgoConstructParams tempInferParams;
  AlgoInferParams inferParams;
  inferParams.dataType = config.inferDataType;
  inferParams.modelPath = config.modelPath;
  inferParams.name = "ocr_reco";
  inferParams.deviceType = config.deviceType;
  inferParams.needDecrypt = config.needDecrypt;
  inferParams.decryptkeyStr = config.decryptkeyStr;
  inferParams.maxOutputBufferSizes = {
      {"output_lengths", 1 * sizeof(int64_t)},
      {"argmax_output", 1 * 128 * sizeof(int32_t) * 1}};
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine = config.engineFactory(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  engine->prettyPrintModelInfos();

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {128, 32, 1};
  framePreprocessArg.dataType = config.preprocDataType;
  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = false;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0.f};
  framePreprocessArg.normVals = {255.f};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = {"x"};
  framePreprocessArg.preprocTaskType = config.preprocTaskType;
  framePreprocessArg.outputLocation = config.bufferLocation;
  preprocParams.setParams(framePreprocessArg);

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageGray;
  cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
  AlgoInput algoInput;
  FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageGray);
  frameInput.inputRoi =
      std::make_shared<cv::Rect>(0, 0, imageGray.cols, imageGray.rows);
  algoInput.setParams(frameInput);

  TensorData modelInput;
  framePreproc->process(algoInput, preprocParams, modelInput);

  std::vector<int64_t> inputLengths = {1};
  TypedBuffer inputLengthsTensor;
  inputLengthsTensor.setCpuData(
      ai_core::DataType::INT64,
      std::vector<uint8_t>(
          reinterpret_cast<const uint8_t *>(inputLengths.data()),
          reinterpret_cast<const uint8_t *>(inputLengths.data()) +
              inputLengths.size() * sizeof(int64_t)));
  modelInput.datas.insert({"input_lengths", inputLengthsTensor});
  modelInput.shapes.insert({"input_lengths", {1}});

  TensorData modelOutput;
  ASSERT_EQ(engine->infer(modelInput, modelOutput), InferErrorCode::SUCCESS);

  AlgoPostprocParams postprocParams;
  GenericPostParams genericPost;
  genericPost.algoType = GenericPostParams::AlogType::OCR_RECO;
  genericPost.outputNames = {"output_lengths", "argmax_output"};
  postprocParams.setParams(genericPost);
  AlgoOutput algoOutput;
  ASSERT_TRUE(ocrPostproc->process(modelOutput, preprocParams, algoOutput,
                                   postprocParams));
  OCRRecoRet *ocrRet = algoOutput.getParams<OCRRecoRet>();
  ASSERT_NE(ocrRet, nullptr);
  ASSERT_EQ(ocrRet->outputs.size(), 9);
}

std::vector<TestConfig> GetTestConfigs() {
  std::vector<TestConfig> configs;
#ifdef WITH_ORT
  configs.push_back({"ort",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "assets/models/cnocr136fc.onnx", DataType::FLOAT32,
                     DataType::FLOAT32, DeviceType::CPU,
                     FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
                     BufferLocation::CPU, false});
#endif
#ifdef WITH_NCNN
#endif
#ifdef WITH_TRT
  configs.push_back({"trt",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<TrtAlgoInference>(p);
                     },
                     "assets/models/cnocr136fc_fp16_dynamic.engine",
                     DataType::FLOAT32, DataType::FLOAT32, DeviceType::CPU,
                     FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
                     BufferLocation::CPU, false});
#endif
  return configs;
}

std::string testNameGenerator(const testing::TestParamInfo<TestConfig> &info) {
  return info.param.testName;
}

INSTANTIATE_TEST_SUITE_P(OCRRecoInferBackends, OCRRecoInferTest,
                         ::testing::ValuesIn(GetTestConfigs()),
                         testNameGenerator);

} // namespace testing_ocr_reco