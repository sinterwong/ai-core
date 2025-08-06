#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/algo_input_types.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/postproc_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "postproc/cv_generic_postproc.hpp"
#include "preproc/frame_prep.hpp"
#include "preproc_base.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace testing_diag_spec {
namespace fs = std::filesystem;

using namespace ai_core;
using namespace ai_core::dnn;

struct TestConfig {
  std::string testName;
  std::string moduleName;

  std::function<std::shared_ptr<AlgoInferEngine>(const std::string &,
                                                 const AlgoInferParams &)>
      engineFactory;

  std::string modelFile;
  DataType inferDataType;
  DataType preprocDataType;
  DeviceType deviceType;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  FramePreprocessArg::FramePreprocType preprocTaskType;
  GenericPostParams::GenericAlgoType postprocType;
  BufferLocation bufferLocation = BufferLocation::CPU;
  bool needDecrypt = false;
};

class DiagSpecInferTest : public ::testing::TestWithParam<TestConfig> {
protected:
  void SetUp() override {
    framePreproc = std::make_shared<FramePreprocess>();
    ASSERT_NE(framePreproc, nullptr);

    segPostproc = std::make_shared<CVGenericPostproc>();
    ASSERT_NE(segPostproc, nullptr);
  }

  fs::path resourceDir = fs::path("assets");
  fs::path confDir = resourceDir / "conf";
  fs::path dataDir = resourceDir / "data";
  fs::path modelDir = resourceDir / "models";

  std::string imagePath = (dataDir / "diag_spec/image.png").string();

  std::shared_ptr<PreprocssBase> framePreproc;
  std::shared_ptr<PostprocssBase> segPostproc;
};

TEST_P(DiagSpecInferTest, Normal) {
  const auto &config = GetParam();

  AlgoInferParams inferParams;
  inferParams.dataType = config.inferDataType;
  inferParams.modelPath = (modelDir / config.modelFile).string();
  inferParams.name = "testName";
  inferParams.deviceType = config.deviceType;
  inferParams.needDecrypt = config.needDecrypt;

  auto engine = config.engineFactory(config.moduleName, inferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {320, 320, 3};
  framePreprocessArg.dataType = config.preprocDataType;
  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = config.inputNames;
  framePreprocessArg.preprocTaskType = config.preprocTaskType;
  framePreprocessArg.outputLocation = config.bufferLocation;
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  GenericPostParams genericParams;
  genericParams.postprocType = config.postprocType;
  genericParams.outputNames = config.outputNames;
  postprocParams.setParams(genericParams);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());
  AlgoInput algoInput;
  FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageRGB);
  frameInput.inputRoi = nullptr;
  algoInput.setParams(frameInput);

  TensorData modelInput;
  framePreproc->process(algoInput, preprocParams, modelInput);

  TensorData modelOutput;
  ASSERT_EQ(engine->infer(modelInput, modelOutput), InferErrorCode::SUCCESS);

  AlgoOutput algoOutput;
  ASSERT_TRUE(segPostproc->process(modelOutput, preprocParams, algoOutput,
                                   postprocParams));

  if (config.postprocType == GenericPostParams::GenericAlgoType::B_DIAG_SPEC) {
    auto *bDiagRet = algoOutput.getParams<BDiagSpecRet>();
  } else if (config.postprocType ==
             GenericPostParams::GenericAlgoType::T_DIAG_SPEC) {
    auto *tDiagRet = algoOutput.getParams<TDiagSpecRet>();
  }
}

std::vector<TestConfig> GetTestConfigs() {
  std::vector<TestConfig> configs;
#ifdef WITH_ORT
  configs.push_back(
      {"ort_b_diag_spec",
       "OrtAlgoInference",
       [](const std::string &moduleName, const AlgoInferParams &p) {
         return std::make_shared<AlgoInferEngine>(moduleName, p);
       },
       "b_diag_spec.onnx",
       DataType::FLOAT32,
       DataType::FLOAT32,
       DeviceType::CPU,
       {"input"},
       {"output", "1454", "1458", "1462", "1463", "1467", "1471", "1475",
        "1479", "1483"},
       FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
       GenericPostParams::GenericAlgoType::B_DIAG_SPEC});

  configs.push_back(
      {"ort_t_diag_spec",
       "OrtAlgoInference",
       [](const std::string &moduleName, const AlgoInferParams &p) {
         return std::make_shared<AlgoInferEngine>(moduleName, p);
       },
       "t_diag_spec.onnx",
       DataType::FLOAT32,
       DataType::FLOAT32,
       DeviceType::CPU,
       {"input"},
       {"output", "1129", "1134", "1140", "1146", "1151", "1160", "1165"},
       FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
       GenericPostParams::GenericAlgoType::T_DIAG_SPEC});
#endif
  return configs;
}

std::string testNameGenerator(const testing::TestParamInfo<TestConfig> &info) {
  return info.param.testName;
}

INSTANTIATE_TEST_SUITE_P(DiagSpecInferBackends, DiagSpecInferTest,
                         ::testing::ValuesIn(GetTestConfigs()),
                         testNameGenerator);

} // namespace testing_diag_spec