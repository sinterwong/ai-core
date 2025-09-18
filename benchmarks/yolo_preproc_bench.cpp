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
#include "ai_core/algo_data_types.hpp"
#include "ai_core/algo_preproc.hpp"
#include "ai_core/tensor_data.hpp"
#include <benchmark/benchmark.h>

#include <opencv2/opencv.hpp>

const static auto getFramePreprocessArg =
    [](ai_core::DataType dataType,
       ai_core::FramePreprocessArg::FramePreprocType preprocTaskType,
       ai_core::BufferLocation outputLocation,
       const std::vector<std::string> &inputNames) {
      ai_core::FramePreprocessArg arg;
      arg.modelInputShape = {640, 640, 3};
      arg.dataType = dataType;
      arg.needResize = true;
      arg.isEqualScale = true;
      arg.pad = {0, 0, 0};
      arg.meanVals = {0, 0, 0};
      arg.normVals = {255.f, 255.f, 255.f};
      arg.hwc2chw = true;
      arg.inputNames = inputNames;
      arg.preprocTaskType = preprocTaskType;
      arg.outputLocation = outputLocation;
      return arg;
    };

static void BM_CPU_FramePreproc_Yolo(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();
  ai_core::AlgoPreprocParams preprocParams;
  ai_core::FramePreprocessArg framePreprocessArg = getFramePreprocessArg(
      ai_core::DataType::FLOAT32,
      ai_core::FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
      ai_core::BufferLocation::CPU, {"images"});
  preprocParams.setParams(framePreprocessArg);

  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageRGB);
  frameInput.inputRoi =
      std::make_shared<cv::Rect>(0, 0, imageRGB.cols, imageRGB.rows);
  input.setParams(frameInput);

  std::shared_ptr<ai_core::RuntimeContext> runtimeContext =
      std::make_shared<ai_core::RuntimeContext>();

  ai_core::TensorData modelInput;

  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    preproc.process(input, preprocParams, modelInput, runtimeContext);
  }
  // =================================================

  for (auto _ : state) {
    preproc.process(input, preprocParams, modelInput, runtimeContext);
  }
}
BENCHMARK(BM_CPU_FramePreproc_Yolo)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

#ifdef WITH_TRT
static void BM_GPU_FramePreproc_Yolo(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();

  ai_core::AlgoPreprocParams preprocParams;
  ai_core::FramePreprocessArg framePreprocessArg = getFramePreprocessArg(
      ai_core::DataType::FLOAT32,
      ai_core::FramePreprocessArg::FramePreprocType::CUDA_GPU_GENERIC,
      ai_core::BufferLocation::GPU_DEVICE, {"images"});
  preprocParams.setParams(framePreprocessArg);

  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageRGB);
  frameInput.inputRoi =
      std::make_shared<cv::Rect>(0, 0, imageRGB.cols, imageRGB.rows);
  input.setParams(frameInput);

  ai_core::TensorData modelInput;

  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    preproc.process(input, preprocParams, modelInput);
  }
  // ===============================================

  for (auto _ : state) {
    preproc.process(input, preprocParams, modelInput);
  }
}
BENCHMARK(BM_GPU_FramePreproc_Yolo)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);
#endif
