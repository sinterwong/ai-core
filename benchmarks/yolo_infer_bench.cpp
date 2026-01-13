/**
 * @file yolo_infer_bench.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/algo_types.hpp"
#include "ai_core/infer_engine_wrapper.hpp"
#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/common_types.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/tensor_data.hpp"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

const static auto get_frame_preprocess_arg =
    [](ai_core::DataType data_type,
       ai_core::FramePreprocessArg::FramePreprocType preproc_task_type,
       ai_core::BufferLocation output_location,
       const std::vector<std::string> &input_names) {
      ai_core::FramePreprocessArg arg;
      arg.model_input_shape = {640, 640, 3};
      arg.data_type = data_type;
      arg.need_resize = true;
      arg.is_equal_scale = true;
      arg.pad = {0, 0, 0};
      arg.mean_vals = {0, 0, 0};
      arg.norm_vals = {255.f, 255.f, 255.f};
      arg.hwc2chw = true;
      arg.input_names = input_names;
      arg.preproc_task_type = preproc_task_type;
      arg.output_location = output_location;
      return arg;
    };

const static auto algo_input = []() {
  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frame_input;
  frame_input.image = std::make_shared<cv::Mat>(image_rgb);
  frame_input.input_roi =
      std::make_shared<cv::Rect>(0, 0, image_rgb.cols, image_rgb.rows);
  input.setParams(frame_input);
  return input;
}();

#ifdef WITH_ORT
static void BM_ORT_CPU_DATA_YoloInfer(benchmark::State &state) {
  ai_core::AlgoInferParams infer_params;
  infer_params.model_path = "assets/models/yolov11n-fp16.onnx";
  infer_params.name = "yolov11n";
  infer_params.device_type = ai_core::DeviceType::CPU;
  infer_params.data_type = ai_core::DataType::FLOAT16;

  ai_core::dnn::AlgoInferEngine engine("OrtAlgoInference", infer_params);
  engine.initialize();

  ai_core::AlgoPreprocParams preproc_params;
  ai_core::FramePreprocessArg frame_preprocess_arg = get_frame_preprocess_arg(
      ai_core::DataType::FLOAT16,
      ai_core::FramePreprocessArg::FramePreprocType::OpencvCpuGeneric,
      ai_core::BufferLocation::CPU, {"images"});
  preproc_params.setParams(frame_preprocess_arg);

  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();

  ai_core::AlgoInput input = algo_input;

  std::shared_ptr<ai_core::RuntimeContext> runtime_context =
      std::make_shared<ai_core::RuntimeContext>();
  ai_core::TensorData model_input;
  preproc.process(input, preproc_params, model_input, runtime_context);

  ai_core::TensorData model_output;
  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    engine.infer(model_input, model_output);
  }
  // =================================================

  for (auto _ : state) {
    engine.infer(model_input, model_output);
  }
}
BENCHMARK(BM_ORT_CPU_DATA_YoloInfer)
    ->Repetitions(3)
    ->Iterations(30)
    ->Unit(benchmark::kMillisecond);
#endif

#ifdef WITH_NCNN
static void BM_NCNN_CPU_DATA_YoloInfer(benchmark::State &state) {
  ai_core::AlgoInferParams infer_params;
  infer_params.model_path = "assets/models/yolov11n.ncnn";
  infer_params.name = "yolov11n";
  infer_params.device_type = ai_core::DeviceType::CPU;
  infer_params.data_type = ai_core::DataType::FLOAT32;

  ai_core::dnn::AlgoInferEngine engine("NCNNAlgoInference", infer_params);
  engine.initialize();

  ai_core::AlgoPreprocParams preproc_params;
  ai_core::FramePreprocessArg frame_preprocess_arg = get_frame_preprocess_arg(
      ai_core::DataType::FLOAT32,
      ai_core::FramePreprocessArg::FramePreprocType::OpencvCpuGeneric,
      ai_core::BufferLocation::CPU, {"in0"});
  preproc_params.setParams(frame_preprocess_arg);

  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();

  std::shared_ptr<ai_core::RuntimeContext> runtime_context =
      std::make_shared<ai_core::RuntimeContext>();

  ai_core::AlgoInput input = algo_input;
  ai_core::TensorData model_input;
  preproc.process(input, preproc_params, model_input, runtime_context);

  ai_core::TensorData model_output;
  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    engine.infer(model_input, model_output);
  }
  // =================================================

  for (auto _ : state) {
    engine.infer(model_input, model_output);
  }
}
BENCHMARK(BM_NCNN_CPU_DATA_YoloInfer)
    ->Repetitions(3)
    ->Iterations(30)
    ->Unit(benchmark::kMillisecond);
#endif
