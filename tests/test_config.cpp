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
  EXPECT_TRUE(pp->is_equal_scale); // Protect camelCase configuration parsing.

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

// Out-of-tree postprocess plugins must be drivable from config. An unknown
// module name is not an error: the parameter family is inferred from the keys
// present, so a plugin only needs registering, not a change to this loader.
TEST(ConfigTest, UnknownPostprocModuleFallsBackToGeneric) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"infer": "OrtAlgoInference", "postproc": "MyPoseDecode"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"outputNames": ["boxes", "kpts"]}}
  })";
  const auto cfg = config::parseAlgoConfig(json, "");
  ASSERT_TRUE(cfg.has_postproc);
  const auto *g = cfg.postproc_params.getParams<GenericPostParams>();
  ASSERT_NE(g, nullptr);
  EXPECT_EQ(g->output_names, (std::vector<std::string>{"boxes", "kpts"}));
}

TEST(ConfigTest, UnknownPostprocModuleInfersAnchorDetFromKeys) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"infer": "OrtAlgoInference", "postproc": "MyDet"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"outputNames": ["out"], "condThre": 0.3, "nmsThre": 0.5}}
  })";
  const auto cfg = config::parseAlgoConfig(json, "");
  const auto *a = cfg.postproc_params.getParams<AnchorDetParams>();
  ASSERT_NE(a, nullptr);
  EXPECT_FLOAT_EQ(a->cond_thre, 0.3f);
  EXPECT_FLOAT_EQ(a->nms_thre, 0.5f);
}

TEST(ConfigTest, UnknownPostprocModuleInfersConfidenceFilterFromKeys) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"infer": "OrtAlgoInference", "postproc": "MySeg"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"outputNames": ["out"], "condThre": 0.4}}
  })";
  const auto cfg = config::parseAlgoConfig(json, "");
  const auto *c = cfg.postproc_params.getParams<ConfidenceFilterParams>();
  ASSERT_NE(c, nullptr);
  EXPECT_FLOAT_EQ(c->cond_thre, 0.4f);
}

// When inference guesses wrong, paramFamily says it outright.
TEST(ConfigTest, ExplicitParamFamilyOverridesInference) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"infer": "OrtAlgoInference", "postproc": "MyThing"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"outputNames": ["out"], "condThre": 0.4,
                         "paramFamily": "generic"}}
  })";
  const auto cfg = config::parseAlgoConfig(json, "");
  EXPECT_NE(cfg.postproc_params.getParams<GenericPostParams>(), nullptr);
}

TEST(ConfigTest, InvalidParamFamilyThrows) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"infer": "OrtAlgoInference", "postproc": "MyThing"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"outputNames": ["out"], "paramFamily": "nope"}}
  })";
  EXPECT_THROW(config::parseAlgoConfig(json, ""), config::ConfigError);
}

// Same story on the preprocess side: a custom preprocess plugin consumes the
// same FramePreprocessArg, so the loader must not gate on the module name.
TEST(ConfigTest, UnknownPreprocModuleIsAccepted) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"preproc": "MyPreproc", "infer": "OrtAlgoInference"},
      "preprocParams": {"inputShape": {"w": 8, "h": 8, "c": 3},
                        "inputFormat": "RGB888", "inputNames": ["x"]},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0}}
  })";
  const auto cfg = config::parseAlgoConfig(json, "");
  ASSERT_TRUE(cfg.has_preproc);
  const auto *arg = cfg.preproc_params.getParams<FramePreprocessArg>();
  ASSERT_NE(arg, nullptr);
  EXPECT_EQ(arg->model_input_format, ImagePixelFormat::RGB888);
}

TEST(ConfigTest, InputFormatDefaultsToBgrAndRejectsGarbage) {
  const char *ok = R"({
    "algorithm": {"name": "x",
      "types": {"preproc": "CpuGenericPreprocess", "infer": "OrtAlgoInference"},
      "preprocParams": {"inputShape": {"w": 8, "h": 8, "c": 3},
                        "inputNames": ["x"]},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0}}
  })";
  const auto cfg = config::parseAlgoConfig(ok, "");
  EXPECT_EQ(
      cfg.preproc_params.getParams<FramePreprocessArg>()->model_input_format,
      ImagePixelFormat::BGR888);

  const char *bad = R"({
    "algorithm": {"name": "x",
      "types": {"preproc": "CpuGenericPreprocess", "infer": "OrtAlgoInference"},
      "preprocParams": {"inputShape": {"w": 8, "h": 8, "c": 3},
                        "inputFormat": "YUV420", "inputNames": ["x"]},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0}}
  })";
  EXPECT_THROW(config::parseAlgoConfig(bad, ""), config::ConfigError);
}

TEST(ConfigTest, KeepClassProbsParsed) {
  const char *json = R"({
    "algorithm": {"name": "x",
      "types": {"infer": "OrtAlgoInference", "postproc": "ArgmaxCls"},
      "inferParams": {"modelPath": "m.onnx", "deviceType": 0, "dataType": 0},
      "postprocParams": {"outputNames": ["out"], "keepClassProbs": true}}
  })";
  const auto cfg = config::parseAlgoConfig(json, "");
  const auto *g = cfg.postproc_params.getParams<GenericPostParams>();
  ASSERT_NE(g, nullptr);
  EXPECT_TRUE(g->keep_class_probs);
}

} // namespace testing_config
