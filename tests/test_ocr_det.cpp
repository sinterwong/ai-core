/**
 * @file test_ocr_det.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/i_infer_engine.hpp"
#include "ai_core/i_postprocess.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/input_types.hpp"
#include "ai_core/opencv_interop.hpp"
#include "ai_core/typed_buffer.hpp"
#include "postproc/semantic_seg.hpp"
#include "preproc/cpu_generic_preprocess.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <functional>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#ifdef WITH_NCNN
#include "ncnn/ncnn_infer.hpp"
#endif

#ifdef WITH_ORT
#include "ort/ort_infer.hpp"
#endif

#ifdef WITH_TRT
#include "trt/trt_infer.hpp"
#endif

namespace testing_ocr_det {
namespace fs = std::filesystem;

using namespace ai_core;
using namespace ai_core::dnn;

struct TestConfig {
  std::string test_name;

  std::function<std::shared_ptr<IInferEnginePlugin>(
      const AlgoConstructParams &)>
      engine_factory;

  std::string model_path;
  DataType infer_data_type;
  DataType preproc_data_type;
  DeviceType device_type;
  std::string input_name;
  BufferLocation buffer_location = BufferLocation::CPU;
  bool need_decrypt = false;
  std::string decryptkey_str = "";
};

class OCRDetInferenceTest : public ::testing::TestWithParam<TestConfig> {
protected:
  void SetUp() override {
    m_m_framePreproc = std::make_shared<CpuGenericPreprocess>();
    ASSERT_NE(m_m_framePreproc, nullptr);

    m_confidenceFilterPostproc = std::make_shared<SemanticSeg>();
    ASSERT_NE(m_confidenceFilterPostproc, nullptr);
  }

  void checkResults(const SegRet *seg_ret) {
    ASSERT_NE(seg_ret, nullptr);
    ASSERT_EQ(seg_ret->cls_to_contours.size(), 1);
    ASSERT_EQ(seg_ret->cls_to_contours.at(1).size(), 28);
  }

  fs::path m_resourceDir = fs::path("assets");
  fs::path m_confDir = m_resourceDir / "conf";
  fs::path m_dataDir = m_resourceDir / "data";

  std::string m_image_path = (m_dataDir / "ocr_det/image.png").string();

  std::shared_ptr<IPreprocessPlugin> m_m_framePreproc;
  std::shared_ptr<IPostprocessPlugin> m_confidenceFilterPostproc;
};

TEST_P(OCRDetInferenceTest, Normal) {
  const auto &config = GetParam();

  AlgoConstructParams temp_infer_params;
  AlgoInferParams infer_params;
  infer_params.data_type = config.infer_data_type;
  infer_params.model_path = config.model_path;
  infer_params.name = "ocr_det";
  infer_params.device_type = config.device_type;
  infer_params.need_decrypt = config.need_decrypt;
  infer_params.decryptkey_str = config.decryptkey_str;
  temp_infer_params.setParam("params", infer_params);

  std::shared_ptr<IInferEnginePlugin> engine =
      config.engine_factory(temp_infer_params);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  engine->prettyPrintModelInfos();

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  std::shared_ptr<RuntimeContext> runtime_context =
      std::make_shared<RuntimeContext>();

  AlgoPreprocParams preproc_params;
  FramePreprocessArg frame_preprocess_arg;
  frame_preprocess_arg.model_input_shape = {512, 512, 3};
  frame_preprocess_arg.data_type = config.preproc_data_type;
  frame_preprocess_arg.need_resize = true;
  frame_preprocess_arg.is_equal_scale = true;
  frame_preprocess_arg.pad = {0, 0, 0};
  frame_preprocess_arg.mean_vals = {123.675f, 116.28f, 103.53f};
  frame_preprocess_arg.norm_vals = {58.395f, 57.12f, 57.375f};
  frame_preprocess_arg.hwc2chw = true;
  frame_preprocess_arg.input_names = {config.input_name};
  frame_preprocess_arg.output_location = config.buffer_location;
  preproc_params.setParams(frame_preprocess_arg);

  AlgoPostprocParams postproc_params;
  ConfidenceFilterParams confidence_filter_params;
  confidence_filter_params.cond_thre = 0.3f;
  confidence_filter_params.output_names = {"sigmoid_0.tmp_0"};
  postproc_params.setParams(confidence_filter_params);

  AlgoInput algo_input;
  FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image);
  frame_input.roi = ai_core::Rect{0, 0, image.cols, image.rows};
  algo_input.setParams(frame_input);

  TensorData model_input;
  m_m_framePreproc->process(algo_input, preproc_params, model_input,
                            runtime_context);

  TensorData model_output;
  ASSERT_EQ(engine->infer(model_input, model_output), InferErrorCode::SUCCESS);

  AlgoOutput algo_output;
  ASSERT_EQ(m_confidenceFilterPostproc->process(model_output, postproc_params,
                                                algo_output, runtime_context),
            InferErrorCode::SUCCESS);

  auto *seg_ret = algo_output.getParams<SegRet>();
  checkResults(seg_ret);

  cv::Mat vis_image = image.clone();
  for (const auto &pair : seg_ret->cls_to_contours) {
    for (const auto &contour : pair.second) {
      std::vector<cv::Point> cv_contour;
      cv_contour.reserve(contour.size());
      for (const auto &pt : contour) {
        cv_contour.push_back(ai_core::interop::toCv(pt));
      }
      cv::drawContours(vis_image,
                       std::vector<std::vector<cv::Point>>{cv_contour}, -1,
                       cv::Scalar(0, 255, 0), 2);
    }
  }
  std::string output_filename = "vis_ocr_det_" + config.test_name + ".png";
  cv::imwrite(output_filename, vis_image);
}

std::vector<TestConfig> getTestConfigs() {
  std::vector<TestConfig> configs;
#ifdef WITH_ORT
  configs.push_back({"ort",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "assets/models/ch_PP_ocr_det.onnx", DataType::FLOAT32,
                     DataType::FLOAT32, DeviceType::CPU, "x",
                     BufferLocation::CPU, false});
#endif
#ifdef WITH_NCNN
#endif
#ifdef WITH_TRT
#endif
  return configs;
}

std::string testNameGenerator(const testing::TestParamInfo<TestConfig> &info) {
  return info.param.test_name;
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OCRDetInferenceTest);

INSTANTIATE_TEST_SUITE_P(OCRDetInferenceBackends, OCRDetInferenceTest,
                         ::testing::ValuesIn(getTestConfigs()),
                         testNameGenerator);

} // namespace testing_ocr_det