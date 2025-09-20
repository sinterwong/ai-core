#include "algo_config_parser.hpp"
#include <ai_core/infer_params_types.hpp>
#include <cstddef>
#include <fstream>
#include <logger.hpp>
#include <map>

namespace ai_core::example::utils {
void AlgoConfigParser::loadAndValidateJson() {
  std::ifstream file(mConfigPath);
  if (!file.is_open()) {
    LOG_ERRORS << "Failed to open config file: " << mConfigPath;
    throw std::runtime_error("Failed to open config file: " + mConfigPath);
  }

  try {
    file >> mRootJson;
  } catch (const nlohmann::json::parse_error &e) {
    LOG_ERRORS << "Failed to parse config JSON: " << e.what();
    throw std::runtime_error("Failed to parse config JSON: " +
                             std::string(e.what()));
  }

  if (!mRootJson.contains("algorithm") || !mRootJson["algorithm"].is_object()) {
    LOG_ERRORS << "Config missing 'algorithms' array or it's not an array.";
    throw std::runtime_error(
        "Config missing 'algorithms' array or not an object.");
  }
}

void AlgoConfigParser::parseCommonParams(const nlohmann::json &algoConfig,
                                         ai_core::AlgoConstructParams &params) {
  params.setParam("moduleName",
                  getOptional<std::string>(algoConfig, "name", ""));

  const auto &types = algoConfig.at("types");
  params.setParam("preprocType",
                  getOptional<std::string>(types, "preproc", ""));
  params.setParam("inferType", getOptional<std::string>(types, "infer", ""));
  params.setParam("postprocType",
                  getOptional<std::string>(types, "postproc", ""));
}

void AlgoConfigParser::parseInferParams(const nlohmann::json &inferJson,
                                        ai_core::AlgoConstructParams &params) {
  ai_core::AlgoInferParams inferParams;
  std::string modelRelPath = inferJson.at("modelPath").get<std::string>();
  // 这里的路径拼接逻辑需要关注
  inferParams.modelPath =
      (std::filesystem::path(mConfigPath).parent_path().parent_path() /
       modelRelPath)
          .string();

  inferParams.deviceType =
      static_cast<ai_core::DeviceType>(inferJson.at("deviceType").get<int>());
  inferParams.dataType =
      static_cast<ai_core::DataType>(inferJson.at("dataType").get<int>());
  inferParams.needDecrypt = inferJson.at("needDecrypt").get<bool>();
  inferParams.name =
      getOptional<std::string>(inferJson, "name", "default_infer_name");
  inferParams.maxOutputBufferSizes = getOptional<std::map<std::string, size_t>>(
      inferJson, "maxOutputBufferSizes", {});
  params.setParam("inferParams", inferParams);
}

void AlgoConfigParser::parsePreprocParams(
    const nlohmann::json &preprocJson, ai_core::AlgoConstructParams &params) {
  const std::string preprocType = params.getParam<std::string>("preprocType");

  if (preprocType == "FramePreprocess") {
    params.setParam("preprocParams",
                    parsePreprocFramePreprocessParams(preprocJson));
  } else {
    LOG_ERRORS << "Unsupported preprocType: " << preprocType;
    throw std::runtime_error("Unsupported preprocType");
  }
}

void AlgoConfigParser::parsePostprocParams(
    const nlohmann::json &postProcJson, ai_core::AlgoConstructParams &params) {

  const auto outputNames =
      postProcJson.at("outputNames").get<std::vector<std::string>>();
  const std::string postprocType = params.getParam<std::string>("postprocType");

  if (postprocType == "AnchorDetPostproc") {
    ai_core::AnchorDetParams anchorDetParams;
    anchorDetParams.algoType = static_cast<ai_core::AnchorDetParams::AlgoType>(
        getOptional<int>(postProcJson, "algoType", 0));
    anchorDetParams.condThre =
        getOptional<float>(postProcJson, "condThre", 0.f);
    anchorDetParams.nmsThre = getOptional<float>(postProcJson, "nmsThre", 0.f);
    anchorDetParams.outputNames = outputNames;
    params.setParam("postprocParams", anchorDetParams);
  } else if (postprocType == "CVGenericPostproc") {
    ai_core::GenericPostParams genericPostParams;
    genericPostParams.algoType =
        static_cast<ai_core::GenericPostParams::AlgoType>(
            postProcJson.at("algoType").get<int>());
    genericPostParams.outputNames = outputNames;
    params.setParam("postprocParams", genericPostParams);
  } else if (postprocType == "ConfidenceFilterPostproc") {
    ai_core::ConfidenceFilterParams confidenceFilterParams;
    confidenceFilterParams.algoType =
        static_cast<ai_core::ConfidenceFilterParams::AlgoType>(
            getOptional<int>(postProcJson, "algoType", 0));
    confidenceFilterParams.condThre =
        getOptional<float>(postProcJson, "condThre", 0.f);
    confidenceFilterParams.outputNames = outputNames;
    params.setParam("postprocParams", confidenceFilterParams);
  } else {
    LOG_ERRORS << "Unsupported postprocType: " << postprocType;
    throw std::runtime_error("Unsupported postprocType");
  }
}

ai_core::FramePreprocessArg AlgoConfigParser::parsePreprocFramePreprocessParams(
    const nlohmann::json &preprocJson) {
  ai_core::FramePreprocessArg arg;

  if (preprocJson.contains("inputShape")) {
    arg.modelInputShape.w = preprocJson["inputShape"].at("w").get<int>();
    arg.modelInputShape.h = preprocJson["inputShape"].at("h").get<int>();
    arg.modelInputShape.c = preprocJson["inputShape"].at("c").get<int>();
  }
  arg.meanVals = getOptional<std::vector<float>>(preprocJson, "mean", {});
  arg.normVals = getOptional<std::vector<float>>(preprocJson, "std", {});
  arg.pad = getOptional<std::vector<int>>(preprocJson, "pad", {});
  arg.hwc2chw = getOptional<bool>(preprocJson, "hwc2chw", false);
  arg.needResize = getOptional<bool>(preprocJson, "needResize", false);
  arg.isEqualScale = getOptional<bool>(preprocJson, "isEqualScale", false);
  arg.dataType = static_cast<ai_core::DataType>(getOptional<int>(
      preprocJson, "dataType", static_cast<int>(ai_core::DataType::FLOAT32)));
  arg.outputLocation = static_cast<ai_core::BufferLocation>(
      getOptional<int>(preprocJson, "bufferLocation",
                       static_cast<int>(ai_core::BufferLocation::CPU)));

  if (preprocJson.contains("preprocTaskType")) {
    arg.preprocTaskType =
        static_cast<ai_core::FramePreprocessArg::FramePreprocType>(
            preprocJson["preprocTaskType"].get<int>());
  }

  arg.inputNames = preprocJson.at("inputNames").get<std::vector<std::string>>();
  return arg;
}

ai_core::GenericPostParams AlgoConfigParser::parsePostprocGenericParams(
    const nlohmann::json &postProcJson) {
  ai_core::GenericPostParams genericPostParams;
  genericPostParams.algoType =
      static_cast<ai_core::GenericPostParams::AlgoType>(
          postProcJson.at("algoType").get<int>());
  genericPostParams.outputNames =
      postProcJson.at("outputNames").get<std::vector<std::string>>();
  return genericPostParams;
}

ai_core::AnchorDetParams AlgoConfigParser::parsePostprocAnchorDetParams(
    const nlohmann::json &postProcJson) {
  ai_core::AnchorDetParams anchorDetParams;
  anchorDetParams.algoType = static_cast<ai_core::AnchorDetParams::AlgoType>(
      getOptional<int>(postProcJson, "algoType", 0));
  anchorDetParams.condThre = getOptional<float>(postProcJson, "condThre", 0.5f);
  anchorDetParams.nmsThre = getOptional<float>(postProcJson, "nmsThre", 0.5f);
  anchorDetParams.outputNames =
      postProcJson.at("outputNames").get<std::vector<std::string>>();
  return anchorDetParams;
}

ai_core::ConfidenceFilterParams
AlgoConfigParser::parsePostprocConfidenceFilterParams(
    const nlohmann::json &postProcJson) {
  ai_core::ConfidenceFilterParams confidenceFilterParams;
  confidenceFilterParams.algoType =
      static_cast<ai_core::ConfidenceFilterParams::AlgoType>(
          getOptional<int>(postProcJson, "algoType", 0));
  confidenceFilterParams.condThre =
      getOptional<float>(postProcJson, "condThre", 0.5f);
  confidenceFilterParams.outputNames =
      postProcJson.at("outputNames").get<std::vector<std::string>>();
  return confidenceFilterParams;
}

AlgoConfigParser::AlgoConfigParser(const std::string &configPath)
    : mConfigPath(configPath) {
  loadAndValidateJson();
}

ai_core::AlgoConstructParams AlgoConfigParser::parse() {
  try {
    if (mRootJson.empty()) {
      loadAndValidateJson();
    }

    const auto &algoConfig = mRootJson["algorithm"];

    // 分块解析
    ai_core::AlgoConstructParams params;
    parseCommonParams(algoConfig, params);

    if (algoConfig.contains("preprocParams")) {
      const auto &preprocJson = algoConfig.at("preprocParams");
      parsePreprocParams(preprocJson, params);
    }
    if (algoConfig.contains("inferParams")) {
      const auto &inferJson = algoConfig.at("inferParams");
      parseInferParams(inferJson, params);
    }
    if (algoConfig.contains("postprocParams")) {
      const auto &postprocJson = algoConfig.at("postprocParams");
      parsePostprocParams(postprocJson, params);
    }

    return params;

  } catch (const nlohmann::json::exception &e) {
    LOG_ERRORS << "JSON parsing error: " << e.what();
    throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
  } catch (const std::exception &e) {
    LOG_ERRORS << "Configuration parsing failed: " << e.what();
    throw;
  }
}
} // namespace ai_core::example::utils