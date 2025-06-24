#include "algo_infer_base.hpp"
#include "algo_registrar.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "infer_types.hpp"
#include "vision_infer.hpp"

namespace testing_algo_infer {
namespace fs = std::filesystem;

using namespace infer;
using namespace infer::dnn;

std::vector<std::string> getImagePathsFromDir(const std::string &dir) {
  std::vector<std::string> rets;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".png" ||
        entry.path().extension() == ".jpg" ||
        entry.path().extension() == ".jpeg") {
      rets.push_back(entry.path().string());
    }
  }
  std::sort(rets.begin(), rets.end());
  return rets;
}

class AlgoInferTest : public ::testing::Test {
protected:
  void SetUp() override { AlgoRegistrar::getInstance(); }
  void TearDown() override {}

  fs::path dataDir = fs::path("data");
};

TEST_F(AlgoInferTest, VisionInferTest) {

  std::string imagePath = (dataDir / "yolov11/image.png").string();

  AlgoPostprocParams postProcparams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.inputShape = {640, 640};
  postProcparams.setParams(anchorDetParams);

  AlgoInferParams inferParams;
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
  inferParams.setParams(yoloParam);

  std::shared_ptr<AlgoInferBase> engine = std::make_shared<vision::VisionInfer>(
      "Yolov11Det", inferParams, postProcparams);
  ASSERT_NE(engine, nullptr);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

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

  AlgoOutput algoOutput;
  ASSERT_EQ(engine->infer(algoInput, algoOutput), InferErrorCode::SUCCESS);

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
  cv::imwrite("vis_yolov11_vision_infer_det.png", visImage);

  engine->terminate();
}

TEST_F(AlgoInferTest, FactoryCreatorTest) {

  std::string imagePath = (dataDir / "yolov11/image.png").string();

  AlgoPostprocParams postProcparams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.inputShape = {640, 640};
  postProcparams.setParams(anchorDetParams);

  AlgoInferParams inferParams;
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
  inferParams.setParams(yoloParam);

  std::string moduleName = "Yolov11Det";
  AlgoConstructParams params;
  params.params = {{"moduleName", moduleName},
                   {"inferParams", inferParams},
                   {"postProcParams", postProcparams}};
  std::shared_ptr<AlgoInferBase> engine =
      AlgoInferFactory::instance().create("VisionInfer", params);
  ASSERT_NE(engine, nullptr);

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

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

  AlgoOutput algoOutput;
  ASSERT_EQ(engine->infer(algoInput, algoOutput), InferErrorCode::SUCCESS);

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
  cv::imwrite("vis_yolov11_factory_creator_det.png", visImage);

  engine->terminate();
}
} // namespace testing_algo_infer
