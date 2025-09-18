#include "ai_core/algo_infer.hpp"
#include "ai_core/algo_input_types.hpp"
#include "ai_core/infer_params_types.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace testing_algo_infer {
namespace fs = std::filesystem;
using namespace ai_core;
using namespace ai_core::dnn;

void CheckResults(const DetRet *detRet) {
  EXPECT_NE(detRet, nullptr);
  ASSERT_EQ(detRet->bboxes.size(), 2);

  const auto &box0 =
      (detRet->bboxes[0].label == 0) ? detRet->bboxes[0] : detRet->bboxes[1];
  const auto &box7 =
      (detRet->bboxes[0].label == 7) ? detRet->bboxes[0] : detRet->bboxes[1];

  EXPECT_EQ(box7.label, 7);
  EXPECT_NEAR(box7.score, 0.54, 1e-2);

  EXPECT_EQ(box0.label, 0);
  EXPECT_NEAR(box0.score, 0.8, 1e-2);
}

TEST(AlgoInferenceTest, YoloDet) {
  fs::path resourceDir = fs::path("assets");
  fs::path dataDir = resourceDir / "data";
  std::string imagePath = (dataDir / "yolov11/image.png").string();

  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoModuleTypes moduleTypes;
  moduleTypes.preprocModule = "FramePreprocess";
  moduleTypes.postprocModule = "AnchorDetPostproc";

#ifdef WITH_ORT
  moduleTypes.inferModule = "OrtAlgoInference";
#elif WITH_NCNN
  moduleTypes.inferModule = "NCNNAlgoInference";
#elif WITH_TRT
  moduleTypes.inferModule = "TrtAlgoInference";
#else
  GTEST_SKIP() << "No inference backend enabled. Skipping test.";
#endif
  AlgoInferParams inferParams;
  inferParams.dataType = DataType::FLOAT16;
  inferParams.modelPath = "assets/models/yolov11n-fp16.onnx";
  inferParams.name = "yolov11n";
  inferParams.deviceType = DeviceType::CPU;
  inferParams.needDecrypt = false;

  AlgoInference algoInf(moduleTypes, inferParams);
  ASSERT_EQ(algoInf.initialize(), InferErrorCode::SUCCESS);

  AlgoPreprocParams preprocParams;
  FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};

#ifdef WITH_ORT
  framePreprocessArg.dataType = DataType::FLOAT16;
#elif WITH_NCNN
  framePreprocessArg.dataType = DataType::FLOAT32;
#elif WITH_TRT
  framePreprocessArg.dataType = DataType::FLOAT32;
#endif

  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = {"images"};
  framePreprocessArg.preprocTaskType =
      FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC;
  framePreprocessArg.outputLocation = BufferLocation::CPU;
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

  AlgoOutput algoOutput;
  ASSERT_EQ(algoInf.infer(algoInput, preprocParams, postprocParams, algoOutput),
            InferErrorCode::SUCCESS);

  auto *detRet = algoOutput.getParams<DetRet>();
  CheckResults(detRet);
}

} // namespace testing_algo_infer
