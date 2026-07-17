#include "algo_config_parser.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/logger.hpp"
#include <cstddef>
#include <fstream>
#include <map>
#include <set>

namespace ai_core::example::utils {
void AlgoConfigParser::loadAndValidateJson() {
  std::ifstream file(mConfigPath);
  if (!file.is_open()) {
    LOG_ERROR_S << "Failed to open config file: " << mConfigPath;
    throw std::runtime_error("Failed to open config file: " + mConfigPath);
  }

  try {
    file >> mRootJson;
  } catch (const nlohmann::json::parse_error &e) {
    LOG_ERROR_S << "Failed to parse config JSON: " << e.what();
    throw std::runtime_error("Failed to parse config JSON: " +
                             std::string(e.what()));
  }

  if (!mRootJson.contains("algorithm") || !mRootJson["algorithm"].is_object()) {
    LOG_ERROR_S << "Config missing 'algorithms' array or it's not an array.";
    throw std::runtime_error(
        "Config missing 'algorithms' array or not an object.");
  }
}

ai_core::AlgoModuleTypes
AlgoConfigParser::parseModuleTypes(const nlohmann::json &algo_config) {
  const auto &types = algo_config.at("types");
  ai_core::AlgoModuleTypes ret;
  ret.preproc_module = getOptional<std::string>(types, "preproc", "");
  ret.infer_module = getOptional<std::string>(types, "infer", "");
  ret.postproc_module = getOptional<std::string>(types, "postproc", "");
  return ret;
}

ai_core::AlgoInferParams
AlgoConfigParser::parseInferParams(const nlohmann::json &infer_json) {
  ai_core::AlgoInferParams infer_params;
  std::string model_rel_path = infer_json.at("modelPath").get<std::string>();
  // 这里的路径拼接逻辑需要关注
  infer_params.model_path =
      (std::filesystem::path(mConfigPath).parent_path().parent_path() /
       model_rel_path)
          .string();

  infer_params.device_type =
      static_cast<ai_core::DeviceType>(infer_json.at("deviceType").get<int>());
  infer_params.data_type =
      static_cast<ai_core::DataType>(infer_json.at("dataType").get<int>());
  infer_params.need_decrypt = infer_json.at("needDecrypt").get<bool>();
  infer_params.name =
      getOptional<std::string>(infer_json, "name", "default_infer_name");
  infer_params.max_output_buffer_sizes =
      getOptional<std::map<std::string, size_t>>(infer_json,
                                                 "maxOutputBufferSizes", {});

  std::string security_key = SECURITY_KEY;
  infer_params.decryptkey_str = security_key;

  return infer_params;
}

ai_core::AlgoPreprocParams
AlgoConfigParser::parsePreprocParams(const nlohmann::json &preproc_json,
                                     const std::string &preproc_type) {
  // All frame preprocessing plugins consume FramePreprocessArg; the plugin
  // name in "types.preproc" selects the implementation.
  static const std::set<std::string> kFramePreprocPlugins = {
      "CpuGenericPreprocess", "CudaGenericPreprocess",
      "FrameWithMaskPreprocess"};

  ai_core::AlgoPreprocParams params;
  if (kFramePreprocPlugins.count(preproc_type)) {
    auto frame_params = parsePreprocFramePreprocessParams(preproc_json);
    params.setParams(frame_params);
  } else {
    LOG_ERROR_S << "Unsupported preprocType: " << preproc_type;
    throw std::runtime_error("Unsupported preprocType");
  }
  return params;
}

ai_core::AlgoPostprocParams
AlgoConfigParser::parsePostprocParams(const nlohmann::json &post_proc_json,
                                      const std::string &postproc_type) {
  ai_core::AlgoPostprocParams params;
  const auto output_names =
      post_proc_json.at("outputNames").get<std::vector<std::string>>();

  // The plugin name in "types.postproc" determines which params family the
  // plugin consumes.
  static const std::set<std::string> kAnchorDetPlugins = {"Yolov11Det",
                                                          "RTMDet", "NanoDet"};
  static const std::set<std::string> kGenericPlugins = {
      "SoftmaxCls", "FprCls", "RawModelOutput", "OCRReco", "UNetDualOutputSeg"};
  static const std::set<std::string> kConfidenceFilterPlugins = {"SemanticSeg"};

  if (kAnchorDetPlugins.count(postproc_type)) {
    ai_core::AnchorDetParams anchor_det_params;
    anchor_det_params.cond_thre =
        getOptional<float>(post_proc_json, "condThre", 0.f);
    anchor_det_params.nms_thre =
        getOptional<float>(post_proc_json, "nmsThre", 0.f);
    anchor_det_params.output_names = output_names;
    params.setParams(anchor_det_params);
  } else if (kGenericPlugins.count(postproc_type)) {
    ai_core::GenericPostParams generic_post_params;
    generic_post_params.output_names = output_names;
    params.setParams(generic_post_params);
  } else if (kConfidenceFilterPlugins.count(postproc_type)) {
    ai_core::ConfidenceFilterParams confidence_filter_params;
    confidence_filter_params.cond_thre =
        getOptional<float>(post_proc_json, "condThre", 0.f);
    confidence_filter_params.output_names = output_names;
    params.setParams(confidence_filter_params);
  } else {
    LOG_ERROR_S << "Unsupported postprocType: " << postproc_type;
    throw std::runtime_error("Unsupported postprocType");
  }
  return params;
}

ai_core::FramePreprocessArg AlgoConfigParser::parsePreprocFramePreprocessParams(
    const nlohmann::json &preproc_json) {
  ai_core::FramePreprocessArg arg;

  if (preproc_json.contains("inputShape")) {
    arg.model_input_shape.w = preproc_json["inputShape"].at("w").get<int>();
    arg.model_input_shape.h = preproc_json["inputShape"].at("h").get<int>();
    arg.model_input_shape.c = preproc_json["inputShape"].at("c").get<int>();
  }
  arg.mean_vals = getOptional<std::vector<float>>(preproc_json, "mean", {});
  arg.norm_vals = getOptional<std::vector<float>>(preproc_json, "std", {});
  arg.pad = getOptional<std::vector<int>>(preproc_json, "pad", {});
  arg.hwc2chw = getOptional<bool>(preproc_json, "hwc2chw", false);
  arg.need_resize = getOptional<bool>(preproc_json, "needResize", false);
  arg.is_equal_scale = getOptional<bool>(preproc_json, "is_equal_scale", false);
  arg.data_type = static_cast<ai_core::DataType>(getOptional<int>(
      preproc_json, "dataType", static_cast<int>(ai_core::DataType::FLOAT32)));
  arg.output_location = static_cast<ai_core::BufferLocation>(
      getOptional<int>(preproc_json, "buffer_location",
                       static_cast<int>(ai_core::BufferLocation::CPU)));

  arg.input_names =
      preproc_json.at("input_names").get<std::vector<std::string>>();
  return arg;
}

ai_core::GenericPostParams AlgoConfigParser::parsePostprocGenericParams(
    const nlohmann::json &post_proc_json) {
  ai_core::GenericPostParams generic_post_params;
  generic_post_params.output_names =
      post_proc_json.at("outputNames").get<std::vector<std::string>>();
  return generic_post_params;
}

ai_core::AnchorDetParams AlgoConfigParser::parsePostprocAnchorDetParams(
    const nlohmann::json &post_proc_json) {
  ai_core::AnchorDetParams anchor_det_params;
  anchor_det_params.cond_thre =
      getOptional<float>(post_proc_json, "condThre", 0.5f);
  anchor_det_params.nms_thre =
      getOptional<float>(post_proc_json, "nmsThre", 0.5f);
  anchor_det_params.output_names =
      post_proc_json.at("outputNames").get<std::vector<std::string>>();
  return anchor_det_params;
}

ai_core::ConfidenceFilterParams
AlgoConfigParser::parsePostprocConfidenceFilterParams(
    const nlohmann::json &post_proc_json) {
  ai_core::ConfidenceFilterParams confidence_filter_params;
  confidence_filter_params.cond_thre =
      getOptional<float>(post_proc_json, "condThre", 0.5f);
  confidence_filter_params.output_names =
      post_proc_json.at("outputNames").get<std::vector<std::string>>();
  return confidence_filter_params;
}

AlgoConfigParser::AlgoConfigParser(const std::string &config_path)
    : mConfigPath(config_path) {
  loadAndValidateJson();
}

AlgoConfigData AlgoConfigParser::parse() {
  AlgoConfigData ret;
  try {
    if (mRootJson.empty()) {
      loadAndValidateJson();
    }

    const auto &algo_config = mRootJson["algorithm"];

    // 分块解析
    ret.moduleName = getOptional<std::string>(algo_config, "name", "");
    ret.modelTypes = parseModuleTypes(algo_config);
    if (algo_config.contains("preproc_params")) {
      const auto &preproc_json = algo_config.at("preproc_params");
      if (ret.modelTypes.preproc_module.empty()) {
        LOG_ERROR_S
            << "preproc module type is empty, but preproc_params is provided";
        throw std::runtime_error(
            "preproc module type is empty, but preproc_params is provided");
      }
      ret.preproc_params =
          parsePreprocParams(preproc_json, ret.modelTypes.preproc_module);
    }
    if (algo_config.contains("inferParams")) {
      const auto &infer_json = algo_config.at("inferParams");
      ret.inferParams = parseInferParams(infer_json);
    }
    if (algo_config.contains("postproc_params")) {
      const auto &postproc_json = algo_config.at("postproc_params");
      if (ret.modelTypes.postproc_module.empty()) {
        LOG_ERROR_S
            << "postproc module type is empty, but postproc_params is provided";
        throw std::runtime_error(
            "postproc module type is empty, but postproc_params is provided");
      }
      ret.postproc_params =
          parsePostprocParams(postproc_json, ret.modelTypes.postproc_module);
    }
    return ret;

  } catch (const nlohmann::json::exception &e) {
    LOG_ERROR_S << "JSON parsing error: " << e.what();
    throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Configuration parsing failed: " << e.what();
    throw;
  }
}
} // namespace ai_core::example::utils