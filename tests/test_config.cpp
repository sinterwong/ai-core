/**
 * @file test_config.cpp
 * @brief Unit tests for ai_core::config: valid parse round-trip and schema
 * validation. Parses in-memory JSON strings — no assets required.
 */
#include "ai_core/config/algo_config.hpp"
#include "gtest/gtest.h"

namespace testing_config {
using namespace ai_core;

const char *kValidDet = R"({
  "algorithm": {
    "name": "yolo-det",
    "types": {"preproc": "CpuGenericPreprocess", "infer": "OrtAlgoInference",
              "postproc": "Yolov11Det"},
    "preprocParams": {
      "inputShape": {"w": 640, "h": 640, "c": 3},
      "mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0],
      "pad": [0, 0, 0], "isEqualScale": true, "needResize": true,
      "dataType": 0, "hwc2chw": true, "inputNames": ["images"]
    },
    "inferParams": {"modelPath": "models/yolo.onnx", "deviceType": 0,
                    "dataType": 0, "needDecrypt": false},
    "postprocParams": {"condThre": 0.5, "nmsThre": 0.45,
                       "outputNames": ["output0"]}
  }
})";

TEST(ConfigTest, ParsesValidDetConfig) {
  auto cfg = config::parseAlgoConfig(kValidDet, "/root");
  EXPECT_EQ(cfg.name, "yolo-det");
  EXPECT_EQ(cfg.module_types.preproc_module, "CpuGenericPreprocess");
  EXPECT_EQ(cfg.module_types.infer_module, "OrtAlgoInference");
  EXPECT_EQ(cfg.module_types.postproc_module, "Yolov11Det");

  // modelPath resolved against model_root.
  EXPECT_EQ(cfg.infer_params.model_path, "/root/models/yolo.onnx");
  EXPECT_EQ(cfg.infer_params.device_type, DeviceType::CPU);
  // Infer name defaults to the algorithm name when unset.
  EXPECT_EQ(cfg.infer_params.name, "yolo-det");

  ASSERT_TRUE(cfg.has_preproc);
  const auto *pp = cfg.preproc_params.getParams<FramePreprocessArg>();
  ASSERT_NE(pp, nullptr);
  EXPECT_EQ(pp->model_input_shape.w, 640);
  EXPECT_EQ(pp->input_names, (std::vector<std::string>{"images"}));
  EXPECT_TRUE(pp->is_equal_scale); // camelCase key parsed (was the v1.3 bug)

  ASSERT_TRUE(cfg.has_postproc);
  const auto *dp = cfg.postproc_params.getParams<AnchorDetParams>();
  ASSERT_NE(dp, nullptr);
  EXPECT_FLOAT_EQ(dp->cond_thre, 0.5f);
  EXPECT_FLOAT_EQ(dp->nms_thre, 0.45f);
}

TEST(ConfigTest, PostprocFamilySelectedByModule) {
  const char *json = R"({
    "algorithm": {
      "name": "seg",
      "types": {"preproc": "CpuGenericPreprocess", "infer": "OrtAlgoInference",
                "postproc": "SemanticSeg"},
      "preprocParams": {"inputShape": {"w": 512, "h": 512, "c": 3},
                        "inputNames": ["x"]},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"condThre": 0.3, "outputNames": ["out"]}
    }
  })";
  auto cfg = config::parseAlgoConfig(json, "");
  EXPECT_NE(cfg.postproc_params.getParams<ConfidenceFilterParams>(), nullptr);
  EXPECT_EQ(cfg.postproc_params.getParams<AnchorDetParams>(), nullptr);
}

TEST(ConfigTest, GenericPostprocNeedsOnlyOutputNames) {
  const char *json = R"({
    "algorithm": {
      "name": "cls",
      "types": {"infer": "OrtAlgoInference", "postproc": "SoftmaxCls"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"outputNames": ["logits"]}
    }
  })";
  auto cfg = config::parseAlgoConfig(json, "");
  EXPECT_FALSE(cfg.has_preproc); // no preproc module/params -> fine
  ASSERT_TRUE(cfg.has_postproc);
  EXPECT_NE(cfg.postproc_params.getParams<GenericPostParams>(), nullptr);
}

// --- validation failures -----------------------------------------------------

TEST(ConfigTest, InvalidJsonThrows) {
  EXPECT_THROW(config::parseAlgoConfig("{ not json", ""), config::ConfigError);
}

TEST(ConfigTest, MissingAlgorithmThrows) {
  EXPECT_THROW(config::parseAlgoConfig(R"({"foo": 1})", ""),
               config::ConfigError);
}

TEST(ConfigTest, MissingInferParamsThrows) {
  const char *json = R"({
    "algorithm": {"name": "x", "types": {"infer": "OrtAlgoInference"}}
  })";
  EXPECT_THROW(config::parseAlgoConfig(json, ""), config::ConfigError);
}

TEST(ConfigTest, BadDeviceTypeThrows) {
  const char *json = R"({
    "algorithm": {"name": "x", "types": {"infer": "OrtAlgoInference"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 9, "dataType": 0}}
  })";
  EXPECT_THROW(config::parseAlgoConfig(json, ""), config::ConfigError);
}

TEST(ConfigTest, MeanStdLengthMismatchThrows) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"preproc": "CpuGenericPreprocess", "infer": "OrtAlgoInference"},
      "preprocParams": {"inputShape": {"w": 8, "h": 8, "c": 3},
                        "mean": [0.0], "std": [1.0, 2.0], "inputNames": ["x"]},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0}}
  })";
  EXPECT_THROW(config::parseAlgoConfig(json, ""), config::ConfigError);
}

TEST(ConfigTest, PreprocParamsWithoutModuleThrows) {
  const char *json = R"({
    "algorithm": {"name": "x", "types": {"infer": "OrtAlgoInference"},
      "preprocParams": {"inputShape": {"w": 8, "h": 8, "c": 3},
                        "inputNames": ["x"]},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0}}
  })";
  EXPECT_THROW(config::parseAlgoConfig(json, ""), config::ConfigError);
}

TEST(ConfigTest, UnknownPostprocModuleThrows) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"infer": "OrtAlgoInference", "postproc": "NopeDet"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"outputNames": ["out"]}}
  })";
  EXPECT_THROW(config::parseAlgoConfig(json, ""), config::ConfigError);
}

} // namespace testing_config
