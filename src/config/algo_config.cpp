/**
 * @file algo_config.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Implementation of the ai_core::config JSON loader with schema
 * validation. Key style is camelCase throughout, matching assets/conf/*.json.
 * @version 1.0
 * @date 2026-07-18
 *
 * @copyright Copyright (c) 2026
 */
#include "ai_core/config/algo_config.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <set>

namespace ai_core::config {
namespace {

using nlohmann::json;

// --- Small validation helpers ------------------------------------------------

[[noreturn]] void fail(const std::string &ctx, const std::string &msg) {
  throw ConfigError("config error at '" + ctx + "': " + msg);
}

const json &requireObject(const json &parent, const std::string &key,
                          const std::string &ctx) {
  if (!parent.contains(key)) {
    fail(ctx, "missing required object '" + key + "'");
  }
  const json &v = parent.at(key);
  if (!v.is_object()) {
    fail(ctx + "." + key, "expected an object");
  }
  return v;
}

template <typename T>
T requireScalar(const json &parent, const std::string &key,
                const std::string &ctx) {
  if (!parent.contains(key)) {
    fail(ctx, "missing required key '" + key + "'");
  }
  try {
    return parent.at(key).get<T>();
  } catch (const json::exception &e) {
    fail(ctx + "." + key, std::string("wrong type: ") + e.what());
  }
}

template <typename T>
T optionalScalar(const json &parent, const std::string &key, T fallback,
                 const std::string &ctx) {
  if (!parent.contains(key)) {
    return fallback;
  }
  try {
    return parent.at(key).get<T>();
  } catch (const json::exception &e) {
    fail(ctx + "." + key, std::string("wrong type: ") + e.what());
  }
}

void requireArrayNonEmpty(const json &parent, const std::string &key,
                          const std::string &ctx) {
  if (!parent.contains(key)) {
    fail(ctx, "missing required array '" + key + "'");
  }
  if (!parent.at(key).is_array() || parent.at(key).empty()) {
    fail(ctx + "." + key, "expected a non-empty array");
  }
}

DeviceType toDeviceType(int v, const std::string &ctx) {
  if (v != 0 && v != 1) {
    fail(ctx + ".deviceType",
         "must be 0 (CPU) or 1 (GPU), got " + std::to_string(v));
  }
  return static_cast<DeviceType>(v);
}

DataType toDataType(int v, const std::string &ctx) {
  if (v < 0 || v > 4) {
    fail(ctx + ".dataType",
         "must be 0..4 (FLOAT32/FLOAT16/INT32/INT64/INT8), got " +
             std::to_string(v));
  }
  return static_cast<DataType>(v);
}

// --- Section parsers ---------------------------------------------------------

AlgoModuleTypes parseModuleTypes(const json &types) {
  AlgoModuleTypes t;
  t.preproc_module = optionalScalar<std::string>(types, "preproc", "", "types");
  t.infer_module = requireScalar<std::string>(types, "infer", "types");
  t.postproc_module =
      optionalScalar<std::string>(types, "postproc", "", "types");
  return t;
}

FramePreprocessArg parseFramePreprocessArg(const json &p) {
  const std::string ctx = "preprocParams";
  FramePreprocessArg arg;

  const json &shape = requireObject(p, "inputShape", ctx);
  arg.model_input_shape.w = requireScalar<int>(shape, "w", ctx + ".inputShape");
  arg.model_input_shape.h = requireScalar<int>(shape, "h", ctx + ".inputShape");
  arg.model_input_shape.c = requireScalar<int>(shape, "c", ctx + ".inputShape");
  if (arg.model_input_shape.w <= 0 || arg.model_input_shape.h <= 0 ||
      arg.model_input_shape.c <= 0) {
    fail(ctx + ".inputShape", "w/h/c must all be positive");
  }

  arg.mean_vals = optionalScalar<std::vector<float>>(p, "mean", {}, ctx);
  arg.norm_vals = optionalScalar<std::vector<float>>(p, "std", {}, ctx);
  if (arg.mean_vals.size() != arg.norm_vals.size()) {
    fail(ctx, "'mean' and 'std' must have the same length");
  }
  arg.pad = optionalScalar<std::vector<int>>(p, "pad", {}, ctx);
  arg.hwc2chw = optionalScalar<bool>(p, "hwc2chw", false, ctx);
  arg.need_resize = optionalScalar<bool>(p, "needResize", false, ctx);
  arg.is_equal_scale = optionalScalar<bool>(p, "isEqualScale", false, ctx);
  arg.data_type =
      toDataType(optionalScalar<int>(p, "dataType",
                                     static_cast<int>(DataType::FLOAT32), ctx),
                 ctx);
  arg.output_location = static_cast<BufferLocation>(optionalScalar<int>(
      p, "bufferLocation", static_cast<int>(BufferLocation::CPU), ctx));

  requireArrayNonEmpty(p, "inputNames", ctx);
  arg.input_names = p.at("inputNames").get<std::vector<std::string>>();
  return arg;
}

AlgoInferParams parseInferParams(const json &p, const std::string &model_root) {
  const std::string ctx = "inferParams";
  AlgoInferParams ip;

  std::string model_rel = requireScalar<std::string>(p, "modelPath", ctx);
  ip.model_path =
      model_root.empty()
          ? model_rel
          : (std::filesystem::path(model_root) / model_rel).string();

  ip.device_type = toDeviceType(requireScalar<int>(p, "deviceType", ctx), ctx);
  ip.data_type = toDataType(requireScalar<int>(p, "dataType", ctx), ctx);
  ip.need_decrypt = optionalScalar<bool>(p, "needDecrypt", false, ctx);
  ip.name = optionalScalar<std::string>(p, "name", "", ctx);
  ip.decryptkey_str = optionalScalar<std::string>(p, "decryptKey", "", ctx);
  ip.max_output_buffer_sizes = optionalScalar<std::map<std::string, size_t>>(
      p, "maxOutputBufferSizes", {}, ctx);
  ip.intra_op_num_threads = optionalScalar<int>(p, "intraOpNumThreads", 0, ctx);
  ip.inter_op_num_threads = optionalScalar<int>(p, "interOpNumThreads", 0, ctx);
  return ip;
}

AlgoPostprocParams parsePostprocParams(const json &p,
                                       const std::string &postproc_module) {
  const std::string ctx = "postprocParams";
  static const std::set<std::string> kAnchorDet = {"Yolov11Det", "RTMDet",
                                                   "NanoDet"};
  static const std::set<std::string> kGeneric = {
      "SoftmaxCls", "FprCls", "RawModelOutput", "OCRReco", "UNetDualOutputSeg"};
  static const std::set<std::string> kConfidenceFilter = {"SemanticSeg"};

  requireArrayNonEmpty(p, "outputNames", ctx);
  auto output_names = p.at("outputNames").get<std::vector<std::string>>();

  AlgoPostprocParams params;
  if (kAnchorDet.count(postproc_module)) {
    AnchorDetParams a;
    a.cond_thre = requireScalar<float>(p, "condThre", ctx);
    a.nms_thre = requireScalar<float>(p, "nmsThre", ctx);
    a.output_names = std::move(output_names);
    params.setParams(a);
  } else if (kConfidenceFilter.count(postproc_module)) {
    ConfidenceFilterParams c;
    c.cond_thre = requireScalar<float>(p, "condThre", ctx);
    c.output_names = std::move(output_names);
    params.setParams(c);
  } else if (kGeneric.count(postproc_module)) {
    GenericPostParams g;
    g.output_names = std::move(output_names);
    params.setParams(g);
  } else {
    fail("types.postproc", "unknown postproc module '" + postproc_module + "'");
  }
  return params;
}

AlgoConfig parseRoot(const json &root, const std::string &model_root) {
  if (!root.contains("algorithm") || !root.at("algorithm").is_object()) {
    fail("<root>", "missing required object 'algorithm'");
  }
  const json &algo = root.at("algorithm");

  AlgoConfig cfg;
  cfg.name = optionalScalar<std::string>(algo, "name", "", "algorithm");
  cfg.module_types =
      parseModuleTypes(requireObject(algo, "types", "algorithm"));
  cfg.infer_params = parseInferParams(
      requireObject(algo, "inferParams", "algorithm"), model_root);
  if (cfg.infer_params.name.empty()) {
    cfg.infer_params.name = cfg.name;
  }

  static const std::set<std::string> kFramePreproc = {
      "CpuGenericPreprocess", "CudaGenericPreprocess",
      "FrameWithMaskPreprocess"};

  const bool has_preproc_json = algo.contains("preprocParams");
  const bool has_preproc_module = !cfg.module_types.preproc_module.empty();
  if (has_preproc_json != has_preproc_module) {
    fail("algorithm", "preprocParams and types.preproc must both be present "
                      "or both absent");
  }
  if (has_preproc_json) {
    if (!kFramePreproc.count(cfg.module_types.preproc_module)) {
      fail("types.preproc",
           "unknown preproc module '" + cfg.module_types.preproc_module + "'");
    }
    cfg.preproc_params.setParams(parseFramePreprocessArg(
        requireObject(algo, "preprocParams", "algorithm")));
    cfg.has_preproc = true;
  }

  const bool has_postproc_json = algo.contains("postprocParams");
  const bool has_postproc_module = !cfg.module_types.postproc_module.empty();
  if (has_postproc_json != has_postproc_module) {
    fail("algorithm", "postprocParams and types.postproc must both be present "
                      "or both absent");
  }
  if (has_postproc_json) {
    cfg.postproc_params =
        parsePostprocParams(requireObject(algo, "postprocParams", "algorithm"),
                            cfg.module_types.postproc_module);
    cfg.has_postproc = true;
  }

  return cfg;
}

} // namespace

AlgoConfig parseAlgoConfig(const std::string &json_text,
                           const std::string &model_root) {
  json root;
  try {
    root = json::parse(json_text);
  } catch (const json::parse_error &e) {
    throw ConfigError(std::string("invalid JSON: ") + e.what());
  }
  return parseRoot(root, model_root);
}

AlgoConfig loadAlgoConfig(const std::string &config_path,
                          const std::string &model_root) {
  std::ifstream file(config_path);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path);
  }
  std::string text((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());

  std::string root = model_root;
  if (root.empty()) {
    // <root>/conf/x.json -> model paths resolve under <root>.
    root =
        std::filesystem::path(config_path).parent_path().parent_path().string();
  }
  return parseAlgoConfig(text, root);
}

} // namespace ai_core::config
