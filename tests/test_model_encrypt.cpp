#include "crypto.hpp"
#include "vision_infer.hpp"
#include "vision_registrar.hpp"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <opencv2/core/hal/interface.h>

namespace fs = std::filesystem;

using namespace infer::encrypt;
using namespace infer;
using namespace infer::dnn;

class ModelEncryptTest : public ::testing::Test {
protected:
  void SetUp() override { vision::VisionRegistrar::getInstance(); }
  void TearDown() override {}

  fs::path dataDir = fs::path("data/yolov11");
  fs::path modelDir = fs::path("models");

  const std::string commitCode = "bea4f2fe1875e12e5abcb4d40f85d99262ed3054";
};

TEST_F(ModelEncryptTest, EncryptDecrypt) {
#ifdef USE_NCNN
  auto nonEncFile = (modelDir / "yolov11n.ncnn.bin").string();
  auto encFile = "yolov11n.enc.ncnn.bin";
#else
  auto nonEncFile = (modelDir / "yolov11n.onnx").string();
  auto encFile = "yolov11n.enc.onnx";
#endif
  ASSERT_TRUE(fs::exists(nonEncFile));

  auto cryptoConfig = Crypto::deriveKeyFromCommit(commitCode);
  Crypto crypto(cryptoConfig);
  std::vector<uchar> encData;

  // encrypt file
  crypto.encryptFile(nonEncFile, encFile);

  // decrypt file
  std::vector<uchar> decData;
  crypto.decryptData(encFile, decData);

  // compare
  std::vector<uchar> nonEncData;
  std::ifstream ifs(nonEncFile, std::ios::binary);
  if (!ifs.is_open()) {
    FAIL() << "Failed to open file: " << nonEncFile;
  }
  ifs.seekg(0, std::ios::end);
  nonEncData.resize(ifs.tellg());
  ifs.seekg(0);
  ifs.read((char *)nonEncData.data(), nonEncData.size());
  ASSERT_EQ(decData.size(), nonEncData.size());
  ASSERT_EQ(memcmp(decData.data(), nonEncData.data(), decData.size()), 0);
  fs::remove(encFile);
}

TEST_F(ModelEncryptTest, LoadEncryptdModel) {
#ifdef USE_NCNN
  auto modelPath = (modelDir / "yolov11n.enc.ncnn").string();
#else
  auto modelPath = (modelDir / "yolov11n-fp16.enc.onnx").string();
#endif
  std::unique_ptr<AlgoInferBase> engine;
  AlgoPostprocParams postProcParams;
  AnchorDetParams anchorDetParams;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.inputShape = {640, 640};
  postProcParams.setParams(anchorDetParams);

  AlgoInferParams inferParams;
  FrameInferParam yoloParam;
  yoloParam.name = "test-yolodet";
  yoloParam.modelPath = modelPath;
  yoloParam.inputShape = {640, 640};
  yoloParam.deviceType = DeviceType::CPU;
  yoloParam.dataType = DataType::FLOAT16;
  yoloParam.needDecrypt = true;
  inferParams.setParams(yoloParam);

  engine = std::make_unique<vision::VisionInfer>("Yolov11Det", inferParams,
                                                 postProcParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread((dataDir / "image.png").string());
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

  AlgoOutput algoOutput;
  ASSERT_EQ(engine->infer(algoInput, algoOutput), InferErrorCode::SUCCESS);

  auto *detRet = algoOutput.getParams<DetRet>();
  ASSERT_NE(detRet, nullptr);
  ASSERT_GT(detRet->bboxes.size(), 0);

  cv::Mat visImage = image.clone();
  for (const auto &bbox : detRet->bboxes) {
    cv::rectangle(visImage, bbox.rect, cv::Scalar(0, 255, 0), 2);
    std::stringstream ss;
    ss << bbox.label << ":" << bbox.score;
    cv::putText(visImage, ss.str(), bbox.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("vis_enc_yolov11_det.png", visImage);

  engine->terminate();
}
