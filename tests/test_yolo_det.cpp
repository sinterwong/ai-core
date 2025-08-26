#include "ai_core/algo_input_types.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/typed_buffer.hpp"
#include "infer_base.hpp"
#include "postproc/anchor_det_postproc.hpp"
#include "postproc_base.hpp"
#include "preproc/frame_prep.hpp"
#include "preproc_base.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <functional>
#include <logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
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

namespace testing_yolo_det {
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
  std::string decryptkeyStr = "689bc3e3bdf1c5f2cff81725011ba7d3c0089b25";
};

class YoloDetInferenceTest : public ::testing::TestWithParam<TestConfig> {
protected:
  void SetUp() override {
    Logger::LogConfig logConfig;
    logConfig.appName = "Yolo-Unit-Test";
    logConfig.logPath = "./logs";
    logConfig.logLevel = LogLevel::INFO;
    logConfig.enableConsole = true;
    logConfig.enableColor = true;
    Logger::instance()->initialize(logConfig);

    framePreproc = std::make_shared<FramePreprocess>();
    ASSERT_NE(framePreproc, nullptr);

    yoloDetPostproc = std::make_shared<AnchorDetPostproc>();
    ASSERT_NE(yoloDetPostproc, nullptr);
  }

  void CheckResults(const DetRet *detRet) {
    ASSERT_NE(detRet, nullptr);
    ASSERT_EQ(detRet->bboxes.size(), 2);

    const auto &box0 =
        (detRet->bboxes[0].label == 0) ? detRet->bboxes[0] : detRet->bboxes[1];
    const auto &box7 =
        (detRet->bboxes[0].label == 7) ? detRet->bboxes[0] : detRet->bboxes[1];

    ASSERT_EQ(box7.label, 7);
    ASSERT_NEAR(box7.score, 0.54, 1e-2);

    ASSERT_EQ(box0.label, 0);
    ASSERT_NEAR(box0.score, 0.8, 1e-2);
  }

  fs::path resourceDir = fs::path("assets");
  fs::path confDir = resourceDir / "conf";
  fs::path dataDir = resourceDir / "data";

  std::string imagePath = (dataDir / "yolov11/image.png").string();

  std::shared_ptr<PreprocssBase> framePreproc;
  std::shared_ptr<PostprocssBase> yoloDetPostproc;
};

TEST_P(YoloDetInferenceTest, Normal) {
  const auto &config = GetParam();

  AlgoConstructParams tempInferParams;
  AlgoInferParams inferParams;
  inferParams.dataType = config.inferDataType;
  inferParams.modelPath = config.modelPath;
  inferParams.name = "yolov11n";
  inferParams.deviceType = config.deviceType;
  inferParams.needDecrypt = config.needDecrypt;
  inferParams.decryptkeyStr = config.decryptkeyStr;
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine = config.engineFactory(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  engine->prettyPrintModelInfos();

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};
  framePreprocessArg.dataType = config.preprocDataType; // 使用 config
  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = {config.inputName};
  framePreprocessArg.preprocTaskType = config.preprocTaskType;
  framePreprocessArg.outputLocation = config.bufferLocation;
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.algoType = AnchorDetParams::AlgoType::YOLO_DET_V11;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.outputNames = {"output0"};
  postprocParams.setParams(anchorDetParams);

  AlgoInput algoInput;
  FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageRGB);
  frameInput.inputRoi =
      std::make_shared<cv::Rect>(0, 0, imageRGB.cols, imageRGB.rows);
  algoInput.setParams(frameInput);

  TensorData modelInput;
  framePreproc->process(algoInput, preprocParams, modelInput);

  TensorData modelOutput;
  ASSERT_EQ(engine->infer(modelInput, modelOutput), InferErrorCode::SUCCESS);

  AlgoOutput algoOutput;
  ASSERT_TRUE(yoloDetPostproc->process(modelOutput, preprocParams, algoOutput,
                                       postprocParams));

  auto *detRet = algoOutput.getParams<DetRet>();
  CheckResults(detRet);

  cv::Mat visImage = image.clone();
  for (const auto &bbox : detRet->bboxes) {
    cv::rectangle(visImage, *bbox.rect, cv::Scalar(0, 255, 0), 2);
    std::stringstream ss;
    ss << bbox.label << ":" << std::fixed << std::setprecision(2) << bbox.score;
    cv::putText(visImage, ss.str(), bbox.rect->tl(), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(0, 0, 255), 2);
  }
  std::string output_filename = "vis_yolodet_" + config.testName + ".png";
  cv::imwrite(output_filename, visImage);
}

TEST_P(YoloDetInferenceTest, MultiThreads) {
  const auto &config = GetParam();

  AlgoConstructParams tempInferParams;
  AlgoInferParams inferParams;
  inferParams.dataType = config.inferDataType;
  inferParams.modelPath = config.modelPath;
  inferParams.name = "yolov11n";
  inferParams.deviceType = config.deviceType;
  inferParams.needDecrypt = config.needDecrypt;
  inferParams.decryptkeyStr = config.decryptkeyStr;
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine = config.engineFactory(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};
  framePreprocessArg.dataType = config.preprocDataType;
  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = {config.inputName};
  framePreprocessArg.preprocTaskType = config.preprocTaskType;
  framePreprocessArg.outputLocation = config.bufferLocation;
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.algoType = AnchorDetParams::AlgoType::YOLO_DET_V11;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.outputNames = {"output0"};
  postprocParams.setParams(anchorDetParams);

  AlgoInput algoInput;
  FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageRGB);
  frameInput.inputRoi =
      std::make_shared<cv::Rect>(0, 0, imageRGB.cols, imageRGB.rows);
  algoInput.setParams(frameInput);

  std::vector<std::thread> threads;
  for (int i = 0; i < 50; ++i) {
    threads.emplace_back([&]() {
      TensorData modelInput;
      framePreproc->process(algoInput, preprocParams, modelInput);

      TensorData modelOutput;
      ASSERT_EQ(engine->infer(modelInput, modelOutput),
                InferErrorCode::SUCCESS);

      AlgoOutput algoOutput;
      ASSERT_TRUE(yoloDetPostproc->process(modelOutput, preprocParams,
                                           algoOutput, postprocParams));

      auto *detRet = algoOutput.getParams<DetRet>();
      CheckResults(detRet);
    });
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

std::vector<TestConfig> GetTestConfigs() {
  std::vector<TestConfig> configs;
#ifdef WITH_ORT
  configs.push_back({"ort",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "assets/models/yolov11n-fp16.onnx", DataType::FLOAT16,
                     DataType::FLOAT16, DeviceType::CPU, "images",
                     FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
                     BufferLocation::CPU, false});
  configs.push_back({"ort_enc",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "assets/enc_models/yolov11n-fp16.enc.onnx",
                     DataType::FLOAT16, DataType::FLOAT16, DeviceType::CPU,
                     "images",
                     FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
                     BufferLocation::CPU, true});
#endif
#ifdef WITH_NCNN
  configs.push_back({"ncnn",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<NCNNAlgoInference>(p);
                     },
                     "assets/models/yolov11n.ncnn", DataType::FLOAT16,
                     DataType::FLOAT32, DeviceType::CPU, "in0",
                     FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
                     BufferLocation::CPU, false});
  configs.push_back({"ncnn_enc",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<NCNNAlgoInference>(p);
                     },
                     "assets/enc_models/yolov11n.enc.ncnn", DataType::FLOAT16,
                     DataType::FLOAT32, DeviceType::CPU, "in0",
                     FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
                     BufferLocation::CPU, true});
#endif
#ifdef WITH_TRT
  configs.push_back({"trt",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<TrtAlgoInference>(p);
                     },
                     "assets/models/yolov11n_trt_fp16.engine",
                     DataType::FLOAT32, DataType::FLOAT32, DeviceType::GPU,
                     "images",
                     FramePreprocessArg::FramePreprocType::CUDA_GPU_GENERIC,
                     BufferLocation::GPU_DEVICE, false});
  configs.push_back({"trt_enc",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<TrtAlgoInference>(p);
                     },
                     "assets/enc_models/yolov11n_trt_fp16.enc.engine",
                     DataType::FLOAT32, DataType::FLOAT32, DeviceType::GPU,
                     "images",
                     FramePreprocessArg::FramePreprocType::CUDA_GPU_GENERIC,
                     BufferLocation::GPU_DEVICE, true});
#endif
  return configs;
}

std::string testNameGenerator(const testing::TestParamInfo<TestConfig> &info) {
  return info.param.testName;
}

INSTANTIATE_TEST_SUITE_P(YoloInferenceBackends, YoloDetInferenceTest,
                         ::testing::ValuesIn(GetTestConfigs()),
                         testNameGenerator);

} // namespace testing_yolo_det