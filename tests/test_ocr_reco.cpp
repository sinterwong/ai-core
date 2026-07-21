/**
 * @file test_ocr_reco.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-25
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/i_infer_engine.hpp"
#include "ai_core/i_postprocess.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/input_types.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/opencv_interop.hpp"
#include "ai_core/typed_buffer.hpp"
#include "postproc/ocr_reco.hpp"
#include "preproc/cpu_generic_preprocess.hpp"
#include "gtest/gtest.h"
#include <cstdint>
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

namespace testing_ocr_reco {
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
  BufferLocation buffer_location = BufferLocation::CPU;
  bool need_decrypt = false;
  std::string decryptkey_str = "";
};

class OCRRecoInferTest : public ::testing::TestWithParam<TestConfig> {
protected:
  void SetUp() override {
    ai_core::logging::Logger::instance().setLevel(
        ai_core::logging::LogLevel::Trace);
    ai_core::logging::Logger::instance().enableConsole(true);
    ai_core::logging::Logger::instance().enableFile(false);
    ai_core::logging::Logger::instance().enableColor(true);
    ai_core::logging::Logger::instance().enableAsync(false);

    m_framePreproc = std::make_shared<CpuGenericPreprocess>();
    ASSERT_NE(m_framePreproc, nullptr);

    m_ocrPostproc = std::make_shared<OCRReco>();
    ASSERT_NE(m_ocrPostproc, nullptr);
  }

  void checkResults(const SegRet *seg_ret) {
    ASSERT_NE(seg_ret, nullptr);
    ASSERT_EQ(seg_ret->cls_to_contours.size(), 1);
    ASSERT_EQ(seg_ret->cls_to_contours.at(1).size(), 28);
  }

  fs::path m_resourceDir = fs::path("assets");
  fs::path m_confDir = m_resourceDir / "conf";
  fs::path m_dataDir = m_resourceDir / "data";

  std::string m_image_path = (m_dataDir / "ocr_reco/image.png").string();

  std::shared_ptr<IPreprocessPlugin> m_framePreproc;
  std::shared_ptr<IPostprocessPlugin> m_ocrPostproc;
};

TEST_P(OCRRecoInferTest, Normal) {
  const auto &config = GetParam();

  AlgoConstructParams temp_infer_params;
  AlgoInferParams infer_params;
  infer_params.data_type = config.infer_data_type;
  infer_params.model_path = config.model_path;
  infer_params.name = "ocr_reco";
  infer_params.device_type = config.device_type;
  infer_params.need_decrypt = config.need_decrypt;
  infer_params.decryptkey_str = config.decryptkey_str;
  infer_params.max_output_buffer_sizes = {
      // 这里不设置output_lengths的最大尺寸用来测试自动分配
      // {"output_lengths", 1 * sizeof(int64_t)},
      {"argmax_output", 1 * 32 * sizeof(int32_t) * 1}};
  temp_infer_params.setParam("params", infer_params);

  std::shared_ptr<IInferEnginePlugin> engine =
      config.engine_factory(temp_infer_params);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);
  engine->prettyPrintModelInfos();

  std::shared_ptr<RuntimeContext> runtime_context =
      std::make_shared<RuntimeContext>();

  AlgoPreprocParams preproc_params;
  FramePreprocessArg frame_preprocess_arg;
  frame_preprocess_arg.model_input_shape = {128, 32, 1};
  frame_preprocess_arg.data_type = config.preproc_data_type;
  frame_preprocess_arg.need_resize = true;
  frame_preprocess_arg.is_equal_scale = false;
  frame_preprocess_arg.pad = {0, 0, 0};
  frame_preprocess_arg.mean_vals = {0.f};
  frame_preprocess_arg.norm_vals = {255.f};
  frame_preprocess_arg.hwc2chw = true;
  frame_preprocess_arg.input_names = {"x"};
  frame_preprocess_arg.output_location = config.buffer_location;
  preproc_params.setParams(frame_preprocess_arg);

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());
  cv::Mat image_gray;
  cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  AlgoInput algo_input;
  FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image_gray);
  frame_input.roi = ai_core::Rect{0, 0, image_gray.cols, image_gray.rows};
  algo_input.setParams(frame_input);

  TensorData model_input;
  m_framePreproc->process(algo_input, preproc_params, model_input,
                          runtime_context);

  std::vector<int64_t> input_lengths = {1};
  TypedBuffer input_lengths_tensor;
  input_lengths_tensor = ai_core::TypedBuffer::createFromCpu(
      ai_core::DataType::INT64,
      std::vector<uint8_t>(
          reinterpret_cast<const uint8_t *>(input_lengths.data()),
          reinterpret_cast<const uint8_t *>(input_lengths.data()) +
              input_lengths.size() * sizeof(int64_t)));
  model_input.set("input_lengths", input_lengths_tensor, {1});

  TensorData model_output;
  ASSERT_EQ(engine->infer(model_input, model_output), InferErrorCode::SUCCESS);

  AlgoPostprocParams postproc_params;
  GenericPostParams generic_post;
  generic_post.output_names = {"output_lengths", "argmax_output"};
  postproc_params.setParams(generic_post);
  AlgoOutput algo_output;
  ASSERT_EQ(m_ocrPostproc->process(model_output, postproc_params, algo_output,
                                   runtime_context),
            InferErrorCode::SUCCESS);
  OCRRecoRet *ocr_ret = algo_output.getParams<OCRRecoRet>();
  ASSERT_NE(ocr_ret, nullptr);
  ASSERT_EQ(ocr_ret->outputs.size(), 9);
}

std::vector<TestConfig> getTestConfigs() {
  std::vector<TestConfig> configs;
#ifdef WITH_ORT
  configs.push_back({"ort",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "assets/models/cnocr136fc.onnx", DataType::FLOAT32,
                     DataType::FLOAT32, DeviceType::CPU, BufferLocation::CPU,
                     false});
#endif
#ifdef WITH_NCNN
#endif
#ifdef WITH_TRT
  configs.push_back({"trt",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<TrtAlgoInference>(p);
                     },
                     "assets/models/cnocr136fc_fp16_dynamic.engine",
                     DataType::FLOAT32, DataType::FLOAT32, DeviceType::CPU,
                     BufferLocation::CPU, false});
#endif
  return configs;
}

std::string testNameGenerator(const testing::TestParamInfo<TestConfig> &info) {
  return info.param.test_name;
}
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OCRRecoInferTest);

INSTANTIATE_TEST_SUITE_P(OCRRecoInferBackends, OCRRecoInferTest,
                         ::testing::ValuesIn(getTestConfigs()),
                         testNameGenerator);

} // namespace testing_ocr_reco