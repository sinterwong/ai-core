#include "algo_manager.hpp"
#include "algo_registrar.hpp"
#include "logger/logger.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

#include <nlohmann/json.hpp>

namespace testing_algo_manager {
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

class AlgoManagerTest : public ::testing::Test {
protected:
  void SetUp() override { AlgoRegistrar::getInstance(); }
  void TearDown() override {}

  fs::path confDir = fs::path("conf");
  fs::path dataDir = fs::path("data");
};

AlgoConstructParams loadParamFromJson(const std::string &configPath) {

  AlgoConstructParams params;

  std::ifstream file(configPath);
  if (!file.is_open()) {
    LOG_ERRORS << "Failed to open config file: " << configPath;
    throw std::runtime_error("Failed to open config file: " + configPath);
  }

  nlohmann::json j;
  try {
    file >> j;
  } catch (const nlohmann::json::parse_error &e) {
    LOG_ERRORS << "Failed to parse config JSON: " << e.what();
    throw std::runtime_error("Failed to parse config JSON: " +
                             std::string(e.what()));
  }

  if (!j.contains("algorithms") || !j["algorithms"].is_array()) {
    LOG_ERRORS << "Config missing 'algorithms' array or it's not an array.";
    throw std::runtime_error(
        "Config missing 'algorithms' array or not an array.");
  }

  if (j["algorithms"].empty()) {
    LOG_ERRORS << "Config 'algorithms' array is empty.";
    throw std::runtime_error("Config 'algorithms' array is empty.");
  }

  // Assuming we are loading parameters for the first algorithm in the list
  const auto &algoConfig = j["algorithms"][0];

  std::string moduleName = algoConfig.at("type").get<std::string>();
  params.setParam("moduleName", moduleName);

  if (algoConfig.contains("inferParams")) {
    AlgoInferParams inferParams;
    const auto &inferJson = algoConfig["inferParams"];
    FrameInferParam frameInferParam;
    frameInferParam.modelPath = inferJson.at("modelPath").get<std::string>();
    if (inferJson.contains("inputShape")) {
      frameInferParam.inputShape.w = inferJson["inputShape"].at("w").get<int>();
      frameInferParam.inputShape.h = inferJson["inputShape"].at("h").get<int>();
    }
    frameInferParam.deviceType =
        static_cast<DeviceType>(inferJson.at("deviceType").get<int>());
    frameInferParam.dataType =
        static_cast<DataType>(inferJson.at("dataType").get<int>());
    inferParams.setParams(frameInferParam);
    params.setParam("inferParams", inferParams);
  }

  if (algoConfig.contains("postProcParams")) {
    AlgoPostprocParams postProcParams;
    const auto &postProcJson = algoConfig["postProcParams"];
    AnchorDetParams anchorDetParams;
    anchorDetParams.condThre = postProcJson.at("condThre").get<float>();
    anchorDetParams.nmsThre = postProcJson.at("nmsThre").get<float>();
    if (postProcJson.contains("inputShape")) {
      anchorDetParams.inputShape.w =
          postProcJson["inputShape"].at("w").get<int>();
      anchorDetParams.inputShape.h =
          postProcJson["inputShape"].at("h").get<int>();
    }
    postProcParams.setParams(anchorDetParams);
    params.setParam("postProcParams", postProcParams);
  }
  return params;
}

TEST_F(AlgoManagerTest, Normal) {

  std::string imagePath = (dataDir / "yolov11/image.png").string();

  AlgoConstructParams params =
      loadParamFromJson((confDir / "test_algo_manager.json").string());
  std::string moduleName = params.getParam<std::string>("moduleName");

  std::shared_ptr<AlgoInferBase> engine =
      AlgoInferFactory::instance().create("VisionInfer", params);
  ASSERT_NE(engine, nullptr);

  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  std::shared_ptr<AlgoManager> manager = std::make_shared<AlgoManager>();
  ASSERT_NE(manager, nullptr);

  ASSERT_EQ(manager->registerAlgo(moduleName, engine), InferErrorCode::SUCCESS);
  ASSERT_TRUE(manager->hasAlgo(moduleName));
  ASSERT_NE(manager->getAlgo(moduleName), nullptr);

  // prepare input data
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

  AlgoOutput managerOutput;
  ASSERT_EQ(manager->infer(moduleName, algoInput, managerOutput),
            InferErrorCode::SUCCESS);

  auto managerDetRet = managerOutput.getParams<DetRet>();
  ASSERT_NE(managerDetRet, nullptr);
  ASSERT_EQ(managerDetRet->bboxes.size(), 2);

  ASSERT_EQ(managerDetRet->bboxes[0].label, 7);
  ASSERT_NEAR(managerDetRet->bboxes[0].score, 0.54, 1e-2);

  ASSERT_EQ(managerDetRet->bboxes[1].label, 0);
  ASSERT_NEAR(managerDetRet->bboxes[1].score, 0.8, 1e-2);

  ASSERT_EQ(manager->unregisterAlgo(moduleName), InferErrorCode::SUCCESS);
  ASSERT_FALSE(manager->hasAlgo(moduleName));
  ASSERT_EQ(manager->getAlgo(moduleName), nullptr);
}
} // namespace testing_algo_manager
