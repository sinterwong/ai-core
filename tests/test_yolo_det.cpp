#include "ai_core/i_infer_engine.hpp"
#include "ai_core/i_postprocess.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/input_types.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/typed_buffer.hpp"
#include "postproc/yolo_det.hpp"
#include "preproc/cpu_generic_preprocess.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <functional>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#ifdef WITH_NCNN
#include "ncnn/ncnn_infer.hpp"
#endif

#ifdef WITH_ORT
#include "ort/ort_infer.hpp"
#endif

namespace testing_yolo_det {
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
  std::string decryptkey_str = "689bc3e3bdf1c5f2cff81725011ba7d3c0089b25";
};

class YoloDetInferenceTest : public ::testing::TestWithParam<TestConfig> {
protected:
  void SetUp() override {
    ai_core::logging::Logger::instance().setLevel(
        ai_core::logging::LogLevel::Trace);
    ai_core::logging::Logger::instance().enableConsole(true);
    ai_core::logging::Logger::instance().enableFile(false);
    ai_core::logging::Logger::instance().enableColor(true);

    m_framePreproc = std::make_shared<CpuGenericPreprocess>();
    ASSERT_NE(m_framePreproc, nullptr);

    m_yoloDetPostproc = std::make_shared<Yolov11Det>();
    ASSERT_NE(m_yoloDetPostproc, nullptr);
  }

  void checkResults(const DetRet *det_ret) {
    ASSERT_NE(det_ret, nullptr);
    ASSERT_EQ(det_ret->bboxes.size(), 1);

    const auto &box0 = (det_ret->bboxes[0].label == 0) ? det_ret->bboxes[0]
                                                       : det_ret->bboxes[1];

    ASSERT_EQ(box0.label, 0);
    ASSERT_NEAR(box0.score, 0.811, 1e-2);
  }

  fs::path m_resourceDir = fs::path("assets");
  fs::path m_confDir = m_resourceDir / "conf";
  fs::path m_dataDir = m_resourceDir / "data";

  std::string m_image_path = (m_dataDir / "yolov11/image.png").string();

  std::shared_ptr<IPreprocessPlugin> m_framePreproc;
  std::shared_ptr<IPostprocessPlugin> m_yoloDetPostproc;
};

TEST_P(YoloDetInferenceTest, Normal) {
  const auto &config = GetParam();

  AlgoConstructParams temp_infer_params;
  AlgoInferParams infer_params;
  infer_params.data_type = config.infer_data_type;
  infer_params.model_path = config.model_path;
  infer_params.name = "yolov11n";
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
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preproc_params;
  FramePreprocessArg frame_preprocess_arg;
  frame_preprocess_arg.model_input_shape = {640, 640, 3};
  frame_preprocess_arg.data_type = config.preproc_data_type; // 使用 config
  frame_preprocess_arg.need_resize = true;
  frame_preprocess_arg.is_equal_scale = true;
  frame_preprocess_arg.pad = {0, 0, 0};
  frame_preprocess_arg.mean_vals = {0, 0, 0};
  frame_preprocess_arg.norm_vals = {255.f, 255.f, 255.f};
  frame_preprocess_arg.hwc2chw = true;
  frame_preprocess_arg.input_names = {config.input_name};
  frame_preprocess_arg.output_location = config.buffer_location;
  preproc_params.setParams(frame_preprocess_arg);

  AlgoPostprocParams postproc_params;
  AnchorDetParams anchor_det_params;
  anchor_det_params.cond_thre = 0.5f;
  anchor_det_params.nms_thre = 0.45f;
  anchor_det_params.output_names = {"output0"};
  postproc_params.setParams(anchor_det_params);

  AlgoInput algo_input;
  FrameInput frame_input;
  frame_input.image = std::make_shared<cv::Mat>(image_rgb);
  frame_input.input_roi =
      std::make_shared<cv::Rect>(2, 2, image_rgb.cols - 4, image_rgb.rows - 4);
  algo_input.setParams(frame_input);

  std::shared_ptr<RuntimeContext> runtime_context =
      std::make_shared<RuntimeContext>();
  TensorData model_input;
  m_framePreproc->process(algo_input, preproc_params, model_input,
                          runtime_context);

  TensorData model_output;
  ASSERT_EQ(engine->infer(model_input, model_output), InferErrorCode::SUCCESS);

  AlgoOutput algo_output;
  ASSERT_EQ(m_yoloDetPostproc->process(model_output, postproc_params,
                                       algo_output, runtime_context),
            InferErrorCode::SUCCESS);

  auto *det_ret = algo_output.getParams<DetRet>();
  checkResults(det_ret);

  cv::Mat vis_image = image.clone();
  for (const auto &bbox : det_ret->bboxes) {
    cv::rectangle(vis_image, bbox.rect, cv::Scalar(0, 255, 0), 2);
    std::stringstream ss;
    ss << bbox.label << ":" << std::fixed << std::setprecision(2) << bbox.score;
    cv::putText(vis_image, ss.str(), bbox.rect.tl(), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(0, 0, 255), 2);
  }
  std::string output_filename = "vis_yolodet_" + config.test_name + ".png";
  cv::imwrite(output_filename, vis_image);
}

TEST_P(YoloDetInferenceTest, MultiThreads) {
  const auto &config = GetParam();

  AlgoConstructParams temp_infer_params;
  AlgoInferParams infer_params;
  infer_params.data_type = config.infer_data_type;
  infer_params.model_path = config.model_path;
  infer_params.name = "yolov11n";
  infer_params.device_type = config.device_type;
  infer_params.need_decrypt = config.need_decrypt;
  infer_params.decryptkey_str = config.decryptkey_str;
  temp_infer_params.setParam("params", infer_params);

  std::shared_ptr<IInferEnginePlugin> engine =
      config.engine_factory(temp_infer_params);
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(m_image_path);
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ASSERT_FALSE(image.empty());

  AlgoPreprocParams preproc_params;
  FramePreprocessArg frame_preprocess_arg;
  frame_preprocess_arg.model_input_shape = {640, 640, 3};
  frame_preprocess_arg.data_type = config.preproc_data_type;
  frame_preprocess_arg.need_resize = true;
  frame_preprocess_arg.is_equal_scale = true;
  frame_preprocess_arg.pad = {0, 0, 0};
  frame_preprocess_arg.mean_vals = {0, 0, 0};
  frame_preprocess_arg.norm_vals = {255.f, 255.f, 255.f};
  frame_preprocess_arg.hwc2chw = true;
  frame_preprocess_arg.input_names = {config.input_name};
  frame_preprocess_arg.output_location = config.buffer_location;
  preproc_params.setParams(frame_preprocess_arg);

  AlgoPostprocParams postproc_params;
  AnchorDetParams anchor_det_params;
  anchor_det_params.cond_thre = 0.5f;
  anchor_det_params.nms_thre = 0.45f;
  anchor_det_params.output_names = {"output0"};
  postproc_params.setParams(anchor_det_params);

  AlgoInput algo_input;
  FrameInput frame_input;
  frame_input.image = std::make_shared<cv::Mat>(image_rgb);
  frame_input.input_roi =
      std::make_shared<cv::Rect>(2, 2, image_rgb.cols - 4, image_rgb.rows - 4);
  algo_input.setParams(frame_input);

  std::vector<std::thread> threads;
  for (int i = 0; i < 50; ++i) {
    threads.emplace_back([&]() {
      std::shared_ptr<RuntimeContext> runtime_context =
          std::make_shared<RuntimeContext>();

      TensorData model_input;
      m_framePreproc->process(algo_input, preproc_params, model_input,
                              runtime_context);

      TensorData model_output;
      ASSERT_EQ(engine->infer(model_input, model_output),
                InferErrorCode::SUCCESS);

      AlgoOutput algo_output;
      ASSERT_EQ(m_yoloDetPostproc->process(model_output, postproc_params,
                                           algo_output, runtime_context),
                InferErrorCode::SUCCESS);

      auto *det_ret = algo_output.getParams<DetRet>();
      checkResults(det_ret);
    });
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

std::vector<TestConfig> getTestConfigs() {
  std::vector<TestConfig> configs;
#ifdef WITH_ORT
  configs.push_back({"ort",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "assets/models/yolov11n-fp16.onnx", DataType::FLOAT16,
                     DataType::FLOAT16, DeviceType::CPU, "images",
                     BufferLocation::CPU, false});
  configs.push_back({"ort_enc",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<OrtAlgoInference>(p);
                     },
                     "assets/enc_models/yolov11n-fp16.enc.onnx",
                     DataType::FLOAT16, DataType::FLOAT16, DeviceType::CPU,
                     "images", BufferLocation::CPU, true});
#endif
#ifdef WITH_NCNN
  configs.push_back({"ncnn",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<NCNNAlgoInference>(p);
                     },
                     "assets/models/yolov11n.ncnn", DataType::FLOAT16,
                     DataType::FLOAT32, DeviceType::CPU, "in0",
                     BufferLocation::CPU, false});
  configs.push_back({"ncnn_enc",
                     [](const AlgoConstructParams &p) {
                       return std::make_shared<NCNNAlgoInference>(p);
                     },
                     "assets/enc_models/yolov11n.enc.ncnn", DataType::FLOAT16,
                     DataType::FLOAT32, DeviceType::CPU, "in0",
                     BufferLocation::CPU, true});
#endif
  return configs;
}

std::string testNameGenerator(const testing::TestParamInfo<TestConfig> &info) {
  return info.param.test_name;
}
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(YoloDetInferenceTest);

INSTANTIATE_TEST_SUITE_P(YoloInferenceBackends, YoloDetInferenceTest,
                         ::testing::ValuesIn(getTestConfigs()),
                         testNameGenerator);

} // namespace testing_yolo_det