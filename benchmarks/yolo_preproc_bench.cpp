/**
 * @file yolo_preproc_bench.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/opencv_interop.hpp"
#include "ai_core/algo_types.hpp"
#include "ai_core/tensor_data.hpp"
#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>

const static auto get_frame_preprocess_arg =
    [](ai_core::DataType data_type, ai_core::BufferLocation output_location,
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
      arg.output_location = output_location;
      return arg;
    };

static void BM_CPU_FramePreproc_Yolo(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("CpuGenericPreprocess");
  ai_core::AlgoPreprocParams preproc_params;
  ai_core::FramePreprocessArg frame_preprocess_arg = get_frame_preprocess_arg(
      ai_core::DataType::FLOAT32, ai_core::BufferLocation::CPU, {"images"});
  preproc_params.setParams(frame_preprocess_arg);
  preproc.initialize(preproc_params);

  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image_rgb);
  frame_input.roi = ai_core::Rect{0, 0, image_rgb.cols, image_rgb.rows};
  input.setParams(frame_input);

  std::shared_ptr<ai_core::RuntimeContext> runtime_context =
      std::make_shared<ai_core::RuntimeContext>();

  ai_core::TensorData model_input;

  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    preproc.process(input, model_input, runtime_context);
  }
  // =================================================

  for (auto _ : state) {
    preproc.process(input, model_input, runtime_context);
  }
}
BENCHMARK(BM_CPU_FramePreproc_Yolo)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

#ifdef WITH_TRT
// ============================= Normal ================================
static void BM_GPU_FramePreproc_Yolo(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("CudaGenericPreprocess");

  ai_core::AlgoPreprocParams preproc_params;
  ai_core::FramePreprocessArg frame_preprocess_arg =
      get_frame_preprocess_arg(ai_core::DataType::FLOAT16,
                               ai_core::BufferLocation::GpuDevice, {"images"});
  preproc_params.setParams(frame_preprocess_arg);
  preproc.initialize(preproc_params);

  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image_rgb);
  frame_input.roi = ai_core::Rect{0, 0, image_rgb.cols, image_rgb.rows};
  input.setParams(frame_input);

  ai_core::TensorData model_input;
  std::shared_ptr<ai_core::RuntimeContext> runtime_context =
      std::make_shared<ai_core::RuntimeContext>();

  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    preproc.process(input, model_input, runtime_context);
  }
  // ===============================================

  for (auto _ : state) {
    preproc.process(input, model_input, runtime_context);
  }
}
BENCHMARK(BM_GPU_FramePreproc_Yolo)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

// ========================= Without HWC2CWH ============================
static void BM_GPU_FramePreproc_No_HWC_Yolo(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("CudaGenericPreprocess");

  ai_core::AlgoPreprocParams preproc_params;
  ai_core::FramePreprocessArg frame_preprocess_arg =
      get_frame_preprocess_arg(ai_core::DataType::FLOAT16,
                               ai_core::BufferLocation::GpuDevice, {"images"});
  frame_preprocess_arg.hwc2chw = false;
  preproc_params.setParams(frame_preprocess_arg);
  preproc.initialize(preproc_params);

  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image_rgb);
  frame_input.roi = ai_core::Rect{0, 0, image_rgb.cols, image_rgb.rows};
  input.setParams(frame_input);

  ai_core::TensorData model_input;

  std::shared_ptr<ai_core::RuntimeContext> runtime_context =
      std::make_shared<ai_core::RuntimeContext>();
  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    preproc.process(input, model_input, runtime_context);
  }
  // ===============================================

  for (auto _ : state) {
    preproc.process(input, model_input, runtime_context);
  }
}
BENCHMARK(BM_GPU_FramePreproc_No_HWC_Yolo)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

// ======================== Without FP16 ===========================
static void BM_GPU_FramePreproc_No_FP16_Yolo(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("CudaGenericPreprocess");

  ai_core::AlgoPreprocParams preproc_params;
  ai_core::FramePreprocessArg frame_preprocess_arg =
      get_frame_preprocess_arg(ai_core::DataType::FLOAT32,
                               ai_core::BufferLocation::GpuDevice, {"images"});
  preproc_params.setParams(frame_preprocess_arg);
  preproc.initialize(preproc_params);

  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image_rgb);
  frame_input.roi = ai_core::Rect{0, 0, image_rgb.cols, image_rgb.rows};
  input.setParams(frame_input);

  std::shared_ptr<ai_core::RuntimeContext> runtime_context =
      std::make_shared<ai_core::RuntimeContext>();

  ai_core::TensorData model_input;

  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    preproc.process(input, model_input, runtime_context);
  }
  // ===============================================

  for (auto _ : state) {
    preproc.process(input, model_input, runtime_context);
  }
}
BENCHMARK(BM_GPU_FramePreproc_No_FP16_Yolo)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

// ======================== Without HWC2CWH and FP16 ===========================
static void BM_GPU_FramePreproc_No_HWC_FP16_Yolo(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("CudaGenericPreprocess");

  ai_core::AlgoPreprocParams preproc_params;
  ai_core::FramePreprocessArg frame_preprocess_arg =
      get_frame_preprocess_arg(ai_core::DataType::FLOAT32,
                               ai_core::BufferLocation::GpuDevice, {"images"});
  frame_preprocess_arg.hwc2chw = false;
  preproc_params.setParams(frame_preprocess_arg);
  preproc.initialize(preproc_params);

  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frame_input;
  frame_input.image = ai_core::interop::viewFromMat(image_rgb);
  frame_input.roi = ai_core::Rect{0, 0, image_rgb.cols, image_rgb.rows};
  input.setParams(frame_input);

  ai_core::TensorData model_input;

  std::shared_ptr<ai_core::RuntimeContext> runtime_context =
      std::make_shared<ai_core::RuntimeContext>();

  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    preproc.process(input, model_input, runtime_context);
  }
  // ===============================================

  for (auto _ : state) {
    preproc.process(input, model_input, runtime_context);
  }
}
BENCHMARK(BM_GPU_FramePreproc_No_HWC_FP16_Yolo)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

#endif
