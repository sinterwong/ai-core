#include "core/frame_infer.hpp"
#include "infer_types.hpp"
#include "yolo_det.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

namespace testing_yolo_det {
namespace fs = std::filesystem;

using namespace infer;
using namespace infer::dnn;
class YoloDetInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {

    AnchorDetParams anchorDetParams;
    anchorDetParams.condThre = 0.5f;
    anchorDetParams.nmsThre = 0.45f;
    anchorDetParams.inputShape = {640, 640};
    params.setParams(anchorDetParams);

    yoloDet = std::make_shared<vision::Yolov11Det>(params);
    ASSERT_NE(yoloDet, nullptr);
  }
  void TearDown() override {}

  fs::path dataDir = fs::path("data");

  std::string imagePath = (dataDir / "yolov11/image.png").string();

  AlgoPostprocParams params;

  std::shared_ptr<vision::VisionBase> yoloDet;
};

TEST_F(YoloDetInferenceTest, Normal) {
  FrameInferParam yoloParam;
  yoloParam.name = "test-yolodet";

#ifdef USE_NCNN
  yoloParam.modelPath = "models/yolov11n.ncnn";
#else
  yoloParam.modelPath = "models/yolov11n-fp16.onnx";
#endif
  yoloParam.inputShape = {640, 640};
  yoloParam.deviceType = DeviceType::CPU;
  yoloParam.dataType = DataType::FLOAT16;

  std::shared_ptr<Inference> engine =
      std::make_shared<FrameInference>(yoloParam);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.args.originShape = {imageRGB.cols, imageRGB.rows};
  frameInput.args.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  frameInput.args.isEqualScale = true;
  frameInput.args.pad = {0, 0, 0};
  frameInput.args.meanVals = {0, 0, 0};
  frameInput.args.normVals = {255.f, 255.f, 255.f};

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  ModelOutput modelOutput;
  ASSERT_EQ(engine->infer(algoInput, modelOutput), InferErrorCode::SUCCESS);

  auto frameInputPtr = algoInput.getParams<FrameInput>();
  AlgoOutput algoOutput;
  ASSERT_TRUE(
      yoloDet->processOutput(modelOutput, frameInputPtr->args, algoOutput));

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
  cv::imwrite("vis_yolodet.png", visImage);

  engine->terminate();
}

TEST_F(YoloDetInferenceTest, MulitThreads) {
  FrameInferParam yoloParam;
  yoloParam.name = "test-yolodet";

#ifdef USE_NCNN
  yoloParam.modelPath = "models/yolov11n.ncnn";
#else
  yoloParam.modelPath = "models/yolov11n-fp16.onnx";
#endif
  yoloParam.inputShape = {640, 640};
  yoloParam.deviceType = DeviceType::CPU;
  yoloParam.dataType = DataType::FLOAT16;

  std::shared_ptr<Inference> engine =
      std::make_shared<FrameInference>(yoloParam);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.args.originShape = {imageRGB.cols, imageRGB.rows};
  frameInput.args.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  frameInput.args.isEqualScale = true;
  frameInput.args.pad = {0, 0, 0};
  frameInput.args.meanVals = {0, 0, 0};
  frameInput.args.normVals = {255.f, 255.f, 255.f};

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  std::vector<std::thread> threads;
  for (int i = 0; i < 100; ++i) {
    threads.emplace_back([&]() {
      ModelOutput modelOutput;
      ASSERT_EQ(engine->infer(algoInput, modelOutput), InferErrorCode::SUCCESS);

      auto frameInputPtr = algoInput.getParams<FrameInput>();
      AlgoOutput algoOutput;
      ASSERT_TRUE(
          yoloDet->processOutput(modelOutput, frameInputPtr->args, algoOutput));

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
  engine->terminate();
}
} // namespace testing_yolo_det