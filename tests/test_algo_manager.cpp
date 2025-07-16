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

  try {
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

    // model name and types
    params.setParam("moduleName", algoConfig["name"].get<std::string>());
    const auto &types = algoConfig["types"];
    params.setParam("preprocType", types["preproc"].get<std::string>());
    params.setParam("inferType", types["infer"].get<std::string>());
    params.setParam("postprocType", types["postproc"].get<std::string>());

    // parse preprocessing args
    const auto &preprocJson = algoConfig["preprocParams"];
    if (params.getParam<std::string>("preprocType") == "FramePreprocess") {
      FramePreprocessArg framePreprocessArg;
      const auto &preprocJson = algoConfig["preprocParams"];
      if (preprocJson.contains("inputShape")) {
        framePreprocessArg.modelInputShape.w =
            preprocJson["inputShape"].at("w").get<int>();
        framePreprocessArg.modelInputShape.h =
            preprocJson["inputShape"].at("h").get<int>();
        framePreprocessArg.modelInputShape.c =
            preprocJson["inputShape"].at("c").get<int>();
      }

      if (preprocJson.contains("mean")) {
        framePreprocessArg.meanVals =
            preprocJson["mean"].get<std::vector<float>>();
      }
      if (preprocJson.contains("std")) {
        framePreprocessArg.normVals =
            preprocJson["std"].get<std::vector<float>>();
      }
      if (preprocJson.contains("pad")) {
        framePreprocessArg.pad = preprocJson["pad"].get<std::vector<int>>();
      }
      if (preprocJson.contains("hwc2chw")) {
        framePreprocessArg.hwc2chw = preprocJson["hwc2chw"].get<bool>();
      } else {
        framePreprocessArg.hwc2chw = false;
      }
      if (preprocJson.contains("needResize")) {
        framePreprocessArg.needResize = preprocJson["needResize"].get<bool>();
      } else {
        framePreprocessArg.needResize = false;
      }

      if (preprocJson.contains("isEqualScale")) {
        framePreprocessArg.isEqualScale =
            preprocJson["isEqualScale"].get<bool>();
      } else {
        framePreprocessArg.isEqualScale = false;
      }
      if (preprocJson.contains("dataType")) {
        framePreprocessArg.dataType =
            static_cast<DataType>(preprocJson["dataType"].get<int>());
      } else {
        framePreprocessArg.dataType = DataType::FLOAT32;
      }
      if (preprocJson.contains("bufferLocation")) {
        framePreprocessArg.outputLocation = static_cast<BufferLocation>(
            preprocJson["bufferLocation"].get<int>());
      } else {
        framePreprocessArg.outputLocation = BufferLocation::CPU;
      }
      if (preprocJson.contains("preprocTaskType")) {
        framePreprocessArg.preprocTaskType =
            static_cast<FramePreprocessArg::FramePreprocType>(
                preprocJson["preprocTaskType"].get<int>());
      }

      framePreprocessArg.inputNames =
          preprocJson["inputNames"].get<std::vector<std::string>>();
      params.setParam("preprocParams", framePreprocessArg);
    } else {
      LOG_ERRORS << "Unsupported preprocType: "
                 << params.getParam<std::string>("preprocType");
      throw std::runtime_error("Unsupported preprocType");
    }

    // parse infer args
    AlgoInferParams inferParams;
    const auto &inferJson = algoConfig["inferParams"];
    std::string modelRelPath = inferJson.at("modelPath").get<std::string>();
    // FIXME: 这里可能是个坑
    inferParams.modelPath =
        (std::filesystem::path(configPath).parent_path().parent_path() /
         modelRelPath)
            .string();
    inferParams.deviceType =
        static_cast<DeviceType>(inferJson.at("deviceType").get<int>());
    inferParams.dataType =
        static_cast<DataType>(inferJson.at("dataType").get<int>());
    inferParams.needDecrypt = inferJson.at("needDecrypt").get<bool>();
    params.setParam("inferParams", inferParams);

    // parse postprocessing args
    const auto &postProcJson = algoConfig["postprocParams"];

    const auto outputNames =
        postProcJson["outputNames"].get<std::vector<std::string>>();
    // FIXME: 就这么先瞎写写吧，后面再完善
    if (params.getParam<std::string>("postprocType") == "RTMDet" ||
        params.getParam<std::string>("postprocType") == "Yolov11Det" ||
        params.getParam<std::string>("postprocType") == "NanoDet") {
      AnchorDetParams anchorDetParams;
      if (postProcJson.contains("condThre")) {
        anchorDetParams.condThre = postProcJson.at("condThre").get<float>();
      }
      if (postProcJson.contains("nmsThre")) {
        anchorDetParams.nmsThre = postProcJson.at("nmsThre").get<float>();
      }
      anchorDetParams.outputNames = outputNames;
      params.setParam("postprocParams", anchorDetParams);
    } else {
      GenericPostParams genericPostParams;
      genericPostParams.outputNames = outputNames;
      params.setParam("postprocParams", genericPostParams);
    }
  } catch (const nlohmann::json::exception &e) {
    LOG_ERRORS << "JSON parsing error: " << e.what();
    throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
  } catch (const std::exception &e) {
    LOG_ERRORS << "Standard exception: " << e.what();
    throw std::runtime_error("Standard exception: " + std::string(e.what()));
  }
  return params;
}

struct ManagerTestParam {
  std::string testName;
  std::string configFile;
};

class AlgoManagerTest : public ::testing::TestWithParam<ManagerTestParam> {
protected:
  void SetUp() override {}
  void TearDown() override {}

  void CheckDetectionResults(const DetRet *detRet) {
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
};

TEST_P(AlgoManagerTest, NormalFlow) {
  const auto &param = GetParam();

  std::string configPath = (confDir / param.configFile).string();
  AlgoConstructParams params = loadParamFromJson(configPath);
  std::string name = params.getParam<std::string>("moduleName");

  AlgoModuleTypes moduleTypes;
  moduleTypes.preprocModule = params.getParam<std::string>("preprocType");
  moduleTypes.inferModule = params.getParam<std::string>("inferType");
  moduleTypes.postprocModule = params.getParam<std::string>("postprocType");

  AlgoInferParams inferParams = params.getParam<AlgoInferParams>("inferParams");
  auto engine = std::make_shared<AlgoInference>(moduleTypes, inferParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto manager = std::make_shared<AlgoManager>();
  ASSERT_NE(manager, nullptr);
  ASSERT_EQ(manager->registerAlgo(name, engine), InferErrorCode::SUCCESS);
  ASSERT_TRUE(manager->hasAlgo(name));
  ASSERT_NE(manager->getAlgo(name), nullptr);

  std::string imagePath = (dataDir / "yolov11/image.png").string();
  cv::Mat image = cv::imread(imagePath);
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preprocParams;
  auto framePreprocessArg =
      params.getParam<FramePreprocessArg>("preprocParams");
  framePreprocessArg.roi =
      std::make_shared<cv::Rect>(0, 0, imageRGB.cols, imageRGB.rows);
  framePreprocessArg.originShape = {imageRGB.cols, imageRGB.rows,
                                    imageRGB.channels()};
  preprocParams.setParams(framePreprocessArg);

  AlgoPostprocParams postprocParams;
  auto anchorDetParams = params.getParam<AnchorDetParams>("postprocParams");
  postprocParams.setParams(anchorDetParams);

  FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageRGB);
  AlgoInput algoInput;
  algoInput.setParams(frameInput);

  AlgoOutput managerOutput;
  ASSERT_EQ(manager->infer(name, algoInput, preprocParams, managerOutput,
                           postprocParams),
            InferErrorCode::SUCCESS);

  auto managerDetRet = managerOutput.getParams<DetRet>();
  CheckDetectionResults(managerDetRet);

  ASSERT_EQ(manager->unregisterAlgo(name), InferErrorCode::SUCCESS);
  ASSERT_FALSE(manager->hasAlgo(name));
  ASSERT_EQ(manager->getAlgo(name), nullptr);
}

std::vector<ManagerTestParam> GetManagerTestParams() {
  std::vector<ManagerTestParam> params;
#ifdef WITH_ORT
  params.push_back({"ORT", "test_algo_manager_ort.json"});
#endif
#ifdef WITH_NCNN
  params.push_back({"NCNN", "test_algo_manager_ncnn.json"});
#endif
#ifdef WITH_TRT
  params.push_back({"TRT", "test_algo_manager_trt.json"});
#endif
  return params;
}

std::string
managerTestNameGenerator(const testing::TestParamInfo<ManagerTestParam> &info) {
  return info.param.testName;
}

INSTANTIATE_TEST_SUITE_P(AlgoManagerBackends, AlgoManagerTest,
                         ::testing::ValuesIn(GetManagerTestParams()),
                         managerTestNameGenerator);
} // namespace testing_algo_manager
