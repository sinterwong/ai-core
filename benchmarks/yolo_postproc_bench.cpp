/**
 * @file yolo_postproc_bench.cpp
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
#include "ai_core/algo_postprocessor.hpp"
#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/tensor_data.hpp"
#include <benchmark/benchmark.h>
#include <memory>
#include <opencv2/opencv.hpp>

#ifdef WITH_ORT
const static auto engine = []() {
  ai_core::AlgoInferParams infer_params;
  infer_params.model_path = "assets/models/yolov11n-fp16.onnx";
  infer_params.name = "yolov11n";
  infer_params.device_type = ai_core::DeviceType::CPU;
  infer_params.data_type = ai_core::DataType::FLOAT16;

  std::shared_ptr<ai_core::dnn::AlgoInferEngine> engine;
  engine = std::make_shared<ai_core::dnn::AlgoInferEngine>("OrtAlgoInference",
                                                           infer_params);
  engine->initialize();
  return engine;
}();
#elif WITH_NCNN
const static auto engine = []() {
  ai_core::AlgoInferParams inferParams;
  inferParams.modelPath = "assets/models/yolov11n.ncnn";
  inferParams.name = "yolov11n";
  inferParams.deviceType = ai_core::DeviceType::CPU;
  inferParams.dataType = ai_core::DataType::FLOAT32;

  std::shared_ptr<ai_core::dnn::AlgoInferEngine> engine;
  engine = std::make_shared<ai_core::dnn::AlgoInferEngine>("NCNNAlgoInference",
                                                           inferParams);
  engine->initialize();
  return engine;
}();
#elif WITH_TRT
const static auto engine = []() {
  ai_core::AlgoInferParams inferParams;
  inferParams.modelPath = "assets/models/yolov11n_trt_fp16.engine";
  inferParams.name = "yolov11n";
  inferParams.deviceType = ai_core::DeviceType::GPU;
  inferParams.dataType = ai_core::DataType::FLOAT32;

  std::shared_ptr<ai_core::dnn::AlgoInferEngine> engine;
  engine = std::make_shared<ai_core::dnn::AlgoInferEngine>("TrtAlgoInference",
                                                           inferParams);
  engine->initialize();
  return engine;
}();
#endif

#if defined(WITH_ORT) || defined(WITH_NCNN) || defined(WITH_TRT)
static void BM_CPU_YoloDetPostproc(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("CpuGenericPreprocess");
  preproc.initialize();

  ai_core::AlgoPreprocParams preproc_params;
  ai_core::FramePreprocessArg frame_preprocess_arg;
  frame_preprocess_arg.model_input_shape = {640, 640, 3};
#ifdef WITH_ORT
  frame_preprocess_arg.data_type = ai_core::DataType::FLOAT16;
  frame_preprocess_arg.input_names = {"images"};
#elif WITH_NCNN
  frame_preprocess_arg_ptr.dataType = ai_core::DataType::FLOAT32;
  frame_preprocess_arg_ptr.input_names = {"in0"};
#elif WITH_TRT
  frame_preprocess_arg_ptr.dataType = ai_core::DataType::FLOAT32;
  frame_preprocess_arg_ptr.input_names = {"images"};
#endif

  frame_preprocess_arg.need_resize = true;
  frame_preprocess_arg.is_equal_scale = true;
  frame_preprocess_arg.pad = {0, 0, 0};
  frame_preprocess_arg.mean_vals = {0, 0, 0};
  frame_preprocess_arg.norm_vals = {255.f, 255.f, 255.f};
  frame_preprocess_arg.hwc2chw = true;
  frame_preprocess_arg.output_location = ai_core::BufferLocation::CPU;
  preproc_params.setParams(frame_preprocess_arg);

  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frame_input;
  frame_input.image = std::make_shared<cv::Mat>(image_rgb);
  frame_input.input_roi =
      std::make_shared<cv::Rect>(0, 0, image_rgb.cols, image_rgb.rows);
  input.setParams(frame_input);

  std::shared_ptr<ai_core::RuntimeContext> runtime_context =
      std::make_shared<ai_core::RuntimeContext>();

  ai_core::TensorData model_input;
  preproc.process(input, preproc_params, model_input, runtime_context);

  ai_core::TensorData model_output;
  engine->infer(model_input, model_output);

  ai_core::dnn::AlgoPostproc postproc("Yolov11Det");
  postproc.initialize();

  ai_core::AlgoPostprocParams postproc_params;
  ai_core::AnchorDetParams anchor_det_params;
  anchor_det_params.cond_thre = 0.5f;
  anchor_det_params.nms_thre = 0.45f;
  anchor_det_params.output_names = {"output0"};
  postproc_params.setParams(anchor_det_params);

  ai_core::AlgoOutput algo_output;

  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    postproc.process(model_output, postproc_params, algo_output, runtime_context);
  }
  // =================================================

  for (auto _ : state) {
    postproc.process(model_output, postproc_params, algo_output, runtime_context);
  }
}
BENCHMARK(BM_CPU_YoloDetPostproc)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);
#endif
