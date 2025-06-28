#include "ai_core/types/algo_input_types.hpp"
#include "ai_core/types/infer_params_types.hpp"
#include "infer_base.hpp"
#include "postproc/yolo_det.hpp"
#include "preproc/frame_prep.hpp"
#include "preproc_base.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#ifdef WITH_NCNN
#include "ncnn/dnn_infer.hpp"
#endif

#ifdef WITH_ORT
#include "ort/dnn_infer.hpp"
#endif
namespace testing_yolo_det {
namespace fs = std::filesystem;

using namespace ai_core;
using namespace ai_core::dnn;
class YoloDetInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    framePreproc = std::make_shared<FramePreprocess>();
    ASSERT_NE(framePreproc, nullptr);

    yoloDetPostproc = std::make_shared<Yolov11Det>();
    ASSERT_NE(yoloDetPostproc, nullptr);
  }
  void TearDown() override {}

  fs::path dataDir = fs::path("data");

  std::string imagePath = (dataDir / "yolov11/image.png").string();

  AlgoInferParams inferParams;

  std::shared_ptr<PreprocssBase> framePreproc;
  std::shared_ptr<PostprocssBase> yoloDetPostproc;
};

#ifdef WITH_ORT
TEST_F(YoloDetInferenceTest, ORTNormal) {
  AlgoConstructParams tempInferParams;
  inferParams.dataType = DataType::FLOAT16;
  inferParams.modelPath = "models/yolov11n-fp16.onnx";
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::CPU;
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine;
  engine = std::make_shared<OrtAlgoInference>(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};
  framePreprocessArg.dataType = DataType::FLOAT16;
  framePreprocessArg.originShape = {imageRGB.cols, imageRGB.rows};
  framePreprocessArg.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  preprocParams.setParams(framePreprocessArg);
  ASSERT_NE(framePreproc, nullptr);
  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.inputName = "images";

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.inputShape = {640, 640};
  postprocParams.setParams(anchorDetParams);

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  TensorData modelInput;
  framePreproc->process(algoInput, preprocParams, modelInput);

  TensorData modelOutput;
  ASSERT_EQ(engine->infer(modelInput, modelOutput), InferErrorCode::SUCCESS);

  auto frameInputPtr = algoInput.getParams<FrameInput>();
  AlgoOutput algoOutput;
  ASSERT_TRUE(yoloDetPostproc->process(modelOutput, preprocParams, algoOutput,
                                       postprocParams));

  auto *detRet = algoOutput.getParams<DetRet>();
  ASSERT_NE(detRet, nullptr);
  ASSERT_EQ(detRet->bboxes.size(), 2);

  ASSERT_EQ(detRet->bboxes[0].label, 7);
  ASSERT_NEAR(detRet->bboxes[0].score, 0.54, 1e-2);

  ASSERT_EQ(detRet->bboxes[1].label, 0);
  ASSERT_NEAR(detRet->bboxes[1].score, 0.8, 1e-2);

  cv::Mat visImage = image.clone();
  for (const auto &bbox : detRet->bboxes) {
    cv::rectangle(visImage, bbox.rect, cv::Scalar(0, 255, 0), 2);
    std::stringstream ss;
    ss << bbox.label << ":" << bbox.score;
    cv::putText(visImage, ss.str(), bbox.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("vis_yolodet_ort.png", visImage);
}

TEST_F(YoloDetInferenceTest, ORTMulitThreads) {

  AlgoConstructParams tempInferParams;
  inferParams.dataType = DataType::FLOAT16;
  inferParams.modelPath = "models/yolov11n-fp16.onnx";
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::CPU;
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine;
  engine = std::make_shared<OrtAlgoInference>(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};
  framePreprocessArg.dataType = DataType::FLOAT16;
  framePreprocessArg.originShape = {imageRGB.cols, imageRGB.rows};
  framePreprocessArg.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  preprocParams.setParams(framePreprocessArg);
  ASSERT_NE(framePreproc, nullptr);
  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.inputName = "images";

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.inputShape = {640, 640};
  postprocParams.setParams(anchorDetParams);

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  std::vector<std::thread> threads;
  for (int i = 0; i < 100; ++i) {
    threads.emplace_back([&]() {
      TensorData modelInput;
      framePreproc->process(algoInput, preprocParams, modelInput);

      TensorData modelOutput;
      ASSERT_EQ(engine->infer(modelInput, modelOutput),
                InferErrorCode::SUCCESS);

      auto frameInputPtr = algoInput.getParams<FrameInput>();
      AlgoOutput algoOutput;
      ASSERT_TRUE(yoloDetPostproc->process(modelOutput, preprocParams,
                                           algoOutput, postprocParams));

      auto *detRet = algoOutput.getParams<DetRet>();
      ASSERT_NE(detRet, nullptr);
      ASSERT_GT(detRet->bboxes.size(), 0);

      ASSERT_EQ(detRet->bboxes[0].label, 7);
      ASSERT_NEAR(detRet->bboxes[0].score, 0.54, 1e-2);

      ASSERT_EQ(detRet->bboxes[1].label, 0);
      ASSERT_NEAR(detRet->bboxes[1].score, 0.8, 1e-2);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}
#endif

#ifdef WITH_NCNN
TEST_F(YoloDetInferenceTest, NCNNNormal) {
  AlgoConstructParams tempInferParams;
  inferParams.dataType = DataType::FLOAT16;
  inferParams.modelPath = "models/yolov11n.ncnn";
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::CPU;
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine;
  engine = std::make_shared<NCNNAlgoInference>(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};
  framePreprocessArg.dataType = DataType::FLOAT32;
  framePreprocessArg.originShape = {imageRGB.cols, imageRGB.rows};
  framePreprocessArg.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  preprocParams.setParams(framePreprocessArg);
  ASSERT_NE(framePreproc, nullptr);
  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.inputName = "in0";

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.inputShape = {640, 640};
  postprocParams.setParams(anchorDetParams);

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  TensorData modelInput;
  framePreproc->process(algoInput, preprocParams, modelInput);

  TensorData modelOutput;
  ASSERT_EQ(engine->infer(modelInput, modelOutput), InferErrorCode::SUCCESS);

  auto frameInputPtr = algoInput.getParams<FrameInput>();
  AlgoOutput algoOutput;
  ASSERT_TRUE(yoloDetPostproc->process(modelOutput, preprocParams, algoOutput,
                                       postprocParams));

  auto *detRet = algoOutput.getParams<DetRet>();
  ASSERT_NE(detRet, nullptr);
  ASSERT_EQ(detRet->bboxes.size(), 2);

  ASSERT_EQ(detRet->bboxes[0].label, 7);
  ASSERT_NEAR(detRet->bboxes[0].score, 0.54, 1e-2);

  ASSERT_EQ(detRet->bboxes[1].label, 0);
  ASSERT_NEAR(detRet->bboxes[1].score, 0.8, 1e-2);

  cv::Mat visImage = image.clone();
  for (const auto &bbox : detRet->bboxes) {
    cv::rectangle(visImage, bbox.rect, cv::Scalar(0, 255, 0), 2);
    std::stringstream ss;
    ss << bbox.label << ":" << bbox.score;
    cv::putText(visImage, ss.str(), bbox.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("vis_yolodet_ncnn.png", visImage);
}

TEST_F(YoloDetInferenceTest, NCNNMultiThreads) {
  AlgoConstructParams tempInferParams;
  inferParams.dataType = DataType::FLOAT16;
  inferParams.modelPath = "models/yolov11n.ncnn";
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::CPU;
  tempInferParams.setParam("params", inferParams);

  std::shared_ptr<InferBase> engine;
  engine = std::make_shared<NCNNAlgoInference>(tempInferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};
  framePreprocessArg.dataType = DataType::FLOAT32;
  framePreprocessArg.originShape = {imageRGB.cols, imageRGB.rows};
  framePreprocessArg.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  preprocParams.setParams(framePreprocessArg);
  ASSERT_NE(framePreproc, nullptr);
  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.inputName = "in0";

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.inputShape = {640, 640};
  postprocParams.setParams(anchorDetParams);

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  std::vector<std::thread> threads;
  for (int i = 0; i < 100; ++i) {
    threads.emplace_back([&]() {
      TensorData modelInput;
      framePreproc->process(algoInput, preprocParams, modelInput);

      TensorData modelOutput;
      ASSERT_EQ(engine->infer(modelInput, modelOutput),
                InferErrorCode::SUCCESS);

      auto frameInputPtr = algoInput.getParams<FrameInput>();
      AlgoOutput algoOutput;
      ASSERT_TRUE(yoloDetPostproc->process(modelOutput, preprocParams,
                                           algoOutput, postprocParams));

      auto *detRet = algoOutput.getParams<DetRet>();
      ASSERT_NE(detRet, nullptr);
      ASSERT_GT(detRet->bboxes.size(), 0);

      ASSERT_EQ(detRet->bboxes[0].label, 7);
      ASSERT_NEAR(detRet->bboxes[0].score, 0.54, 1e-2);

      ASSERT_EQ(detRet->bboxes[1].label, 0);
      ASSERT_NEAR(detRet->bboxes[1].score, 0.8, 1e-2);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}
#endif
} // namespace testing_yolo_det