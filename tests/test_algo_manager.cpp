#include "ai_core/algo_manager.hpp"
#include "logger.hpp"
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

using namespace ai_core;
using namespace ai_core::dnn;

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
  void SetUp() override {}
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

  const auto &algoConfig = j["algorithms"][0];

  if (algoConfig.contains("name")) {
    params.setParam("moduleName", algoConfig["name"].get<std::string>());
  }

  if (algoConfig.contains("types")) {
    const auto &types = algoConfig["types"];
    params.setParam("preprocType", types["preproc"].get<std::string>());
    params.setParam("inferType", types["infer"].get<std::string>());
    params.setParam("postprocType", types["postproc"].get<std::string>());
  }

  if (algoConfig.contains("preprocParams")) {
    FramePreprocessArg framePreprocessArg;
    const auto &preprocJson = algoConfig["preprocParams"];
    framePreprocessArg.modelInputShape.w =
        preprocJson["inputShape"].at("w").get<int>();
    framePreprocessArg.modelInputShape.h =
        preprocJson["inputShape"].at("h").get<int>();
    framePreprocessArg.modelInputShape.c =
        preprocJson["inputShape"].at("c").get<int>();

    framePreprocessArg.meanVals = preprocJson["mean"].get<std::vector<float>>();
    framePreprocessArg.normVals = preprocJson["std"].get<std::vector<float>>();
    framePreprocessArg.dataType =
        static_cast<DataType>(preprocJson["dataType"].get<int>());
    framePreprocessArg.isEqualScale = preprocJson["isEqualScale"].get<bool>();
    framePreprocessArg.needResize = preprocJson["needResize"].get<bool>();
    const auto &padVec = preprocJson["pad"].get<std::vector<int>>();
    framePreprocessArg.pad = cv::Scalar{static_cast<double>(padVec[0]),
                                        static_cast<double>(padVec[1]),
                                        static_cast<double>(padVec[2])};
    framePreprocessArg.hwc2chw = preprocJson["hwc2chw"].get<bool>();
    params.setParam("preprocParams", framePreprocessArg);
  }

  if (algoConfig.contains("inferParams")) {
    AlgoInferParams inferParams;
    const auto &inferJson = algoConfig["inferParams"];
    inferParams.modelPath = inferJson.at("modelPath").get<std::string>();
    inferParams.deviceType =
        static_cast<DeviceType>(inferJson.at("deviceType").get<int>());
    inferParams.dataType =
        static_cast<DataType>(inferJson.at("dataType").get<int>());
    params.setParam("inferParams", inferParams);
  }

  if (algoConfig.contains("postprocParams")) {
    const auto &postProcJson = algoConfig["postprocParams"];
    AnchorDetParams anchorDetParams;
    anchorDetParams.condThre = postProcJson.at("condThre").get<float>();
    anchorDetParams.nmsThre = postProcJson.at("nmsThre").get<float>();
    if (postProcJson.contains("inputShape")) {
      anchorDetParams.inputShape.w =
          postProcJson["inputShape"].at("w").get<int>();
      anchorDetParams.inputShape.h =
          postProcJson["inputShape"].at("h").get<int>();
    }
    params.setParam("postprocParams", anchorDetParams);
  }
  return params;
}

#ifdef WITH_ORT
TEST_F(AlgoManagerTest, ORTNormal) {
  std::string imagePath = (dataDir / "yolov11/image.png").string();
  AlgoConstructParams params =
      loadParamFromJson((confDir / "test_algo_manager_ort.json").string());
  std::string name = params.getParam<std::string>("moduleName");

  AlgoModuleTypes moduleTypes;
  moduleTypes.preprocModule = params.getParam<std::string>("preprocType");
  moduleTypes.inferModule = params.getParam<std::string>("inferType");
  moduleTypes.postprocModule = params.getParam<std::string>("postprocType");

  AlgoInferParams inferParams = params.getParam<AlgoInferParams>("inferParams");

  std::shared_ptr<AlgoInference> engine =
      std::make_shared<AlgoInference>(moduleTypes, inferParams);

  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  std::shared_ptr<AlgoManager> manager = std::make_shared<AlgoManager>();
  ASSERT_NE(manager, nullptr);

  ASSERT_EQ(manager->registerAlgo(name, engine), InferErrorCode::SUCCESS);
  ASSERT_TRUE(manager->hasAlgo(name));
  ASSERT_NE(manager->getAlgo(name), nullptr);

  // prepare input data
  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  auto framePreprocessArg =
      params.getParam<FramePreprocessArg>("preprocParams");
  framePreprocessArg.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  framePreprocessArg.originShape = {imageRGB.cols, imageRGB.rows,
                                    imageRGB.channels()};
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams =
      params.getParam<AnchorDetParams>("postprocParams");
  postprocParams.setParams(anchorDetParams);

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.inputName = "images";

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  AlgoOutput managerOutput;
  ASSERT_EQ(manager->infer(name, algoInput, preprocParams, managerOutput,
                           postprocParams),
            InferErrorCode::SUCCESS);

  auto managerDetRet = managerOutput.getParams<DetRet>();
  ASSERT_NE(managerDetRet, nullptr);
  ASSERT_EQ(managerDetRet->bboxes.size(), 2);

  ASSERT_EQ(managerDetRet->bboxes[0].label, 7);
  ASSERT_NEAR(managerDetRet->bboxes[0].score, 0.54, 1e-2);

  ASSERT_EQ(managerDetRet->bboxes[1].label, 0);
  ASSERT_NEAR(managerDetRet->bboxes[1].score, 0.8, 1e-2);

  ASSERT_EQ(manager->unregisterAlgo(name), InferErrorCode::SUCCESS);
  ASSERT_FALSE(manager->hasAlgo(name));
  ASSERT_EQ(manager->getAlgo(name), nullptr);
}
#endif

#ifdef WITH_NCNN
TEST_F(AlgoManagerTest, NCNNNormal) {
  std::string imagePath = (dataDir / "yolov11/image.png").string();

  AlgoConstructParams params =
      loadParamFromJson((confDir / "test_algo_manager_ncnn.json").string());

  std::string name = params.getParam<std::string>("moduleName");

  AlgoModuleTypes moduleTypes;
  moduleTypes.preprocModule = params.getParam<std::string>("preprocType");
  moduleTypes.inferModule = params.getParam<std::string>("inferType");
  moduleTypes.postprocModule = params.getParam<std::string>("postprocType");

  AlgoInferParams inferParams = params.getParam<AlgoInferParams>("inferParams");

  std::shared_ptr<AlgoInference> engine =
      std::make_shared<AlgoInference>(moduleTypes, inferParams);

  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  std::shared_ptr<AlgoManager> manager = std::make_shared<AlgoManager>();
  ASSERT_NE(manager, nullptr);

  ASSERT_EQ(manager->registerAlgo(name, engine), InferErrorCode::SUCCESS);
  ASSERT_TRUE(manager->hasAlgo(name));
  ASSERT_NE(manager->getAlgo(name), nullptr);

  // prepare input data
  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  auto framePreprocessArg =
      params.getParam<FramePreprocessArg>("preprocParams");
  framePreprocessArg.roi = {0, 0, imageRGB.cols, imageRGB.rows};
  framePreprocessArg.originShape = {imageRGB.cols, imageRGB.rows,
                                    imageRGB.channels()};
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  AnchorDetParams anchorDetParams =
      params.getParam<AnchorDetParams>("postprocParams");
  postprocParams.setParams(anchorDetParams);

  FrameInput frameInput;
  frameInput.image = imageRGB;
  frameInput.inputName = "in0";

  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  AlgoOutput managerOutput;
  ASSERT_EQ(manager->infer(name, algoInput, preprocParams, managerOutput,
                           postprocParams),
            InferErrorCode::SUCCESS);

  auto managerDetRet = managerOutput.getParams<DetRet>();
  ASSERT_NE(managerDetRet, nullptr);
  ASSERT_EQ(managerDetRet->bboxes.size(), 2);

  ASSERT_EQ(managerDetRet->bboxes[0].label, 7);
  ASSERT_NEAR(managerDetRet->bboxes[0].score, 0.54, 1e-2);

  ASSERT_EQ(managerDetRet->bboxes[1].label, 0);
  ASSERT_NEAR(managerDetRet->bboxes[1].score, 0.8, 1e-2);

  ASSERT_EQ(manager->unregisterAlgo(name), InferErrorCode::SUCCESS);
  ASSERT_FALSE(manager->hasAlgo(name));
  ASSERT_EQ(manager->getAlgo(name), nullptr);
}
#endif
} // namespace testing_algo_manager
