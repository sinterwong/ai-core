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
#include "ai_core/algo_data_types.hpp"
#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/algo_preproc.hpp"
#include "ai_core/infer_common_types.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/tensor_data.hpp"
#include <benchmark/benchmark.h>
#include <logger.hpp>
#include <opencv2/opencv.hpp>

// init log
const static auto tempInitLog = []() {
  Logger::LogConfig logConfig;
  logConfig.appName = "YoloInferBenchMark";
  logConfig.logPath = "./yolo_infer_bench_mark_logs";
  logConfig.logLevel = LogLevel::WARNING;
  logConfig.enableConsole = true;
  logConfig.enableColor = true;
  Logger::instance()->initialize(logConfig);
  return true;
}();

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

const static auto algoInput = []() {
  ai_core::AlgoInput input;
  cv::Mat image = cv::imread("assets/data/yolov11/image.png");
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
  ai_core::FrameInput frameInput;
  frameInput.image = std::make_shared<cv::Mat>(imageRGB);
  frameInput.inputRoi =
      std::make_shared<cv::Rect>(0, 0, imageRGB.cols, imageRGB.rows);
  input.setParams(frameInput);
  return input;
}();

#ifdef WITH_ORT
static void BM_ORT_CPU_DATA_YoloInfer(benchmark::State &state) {
  ai_core::AlgoInferParams inferParams;
  inferParams.modelPath = "assets/models/yolov11n-fp16.onnx";
  inferParams.name = "yolov11n";
  inferParams.deviceType = ai_core::DeviceType::CPU;
  inferParams.dataType = ai_core::DataType::FLOAT16;

  ai_core::dnn::AlgoInferEngine engine("OrtAlgoInference", inferParams);
  engine.initialize();

  ai_core::AlgoPreprocParams preprocParams;
  ai_core::FramePreprocessArg framePreprocessArg = getFramePreprocessArg(
      ai_core::DataType::FLOAT16,
      ai_core::FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
      ai_core::BufferLocation::CPU, {"images"});
  preprocParams.setParams(framePreprocessArg);

  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();

  ai_core::AlgoInput input = algoInput;

  std::shared_ptr<ai_core::RuntimeContext> runtimeContext =
      std::make_shared<ai_core::RuntimeContext>();
  ai_core::TensorData modelInput;
  preproc.process(input, preprocParams, modelInput, runtimeContext);

  ai_core::TensorData modelOutput;
  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    engine.infer(modelInput, modelOutput);
  }
  // =================================================

  for (auto _ : state) {
    engine.infer(modelInput, modelOutput);
  }
}
BENCHMARK(BM_ORT_CPU_DATA_YoloInfer)
    ->Repetitions(3)
    ->Iterations(30)
    ->Unit(benchmark::kMillisecond);
#endif

#ifdef WITH_NCNN
static void BM_NCNN_CPU_DATA_YoloInfer(benchmark::State &state) {
  ai_core::AlgoInferParams inferParams;
  inferParams.modelPath = "assets/models/yolov11n.ncnn";
  inferParams.name = "yolov11n";
  inferParams.deviceType = ai_core::DeviceType::CPU;
  inferParams.dataType = ai_core::DataType::FLOAT32;

  ai_core::dnn::AlgoInferEngine engine("NCNNAlgoInference", inferParams);
  engine.initialize();

  ai_core::AlgoPreprocParams preprocParams;
  ai_core::FramePreprocessArg framePreprocessArg = getFramePreprocessArg(
      ai_core::DataType::FLOAT32,
      ai_core::FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
      ai_core::BufferLocation::CPU, {"in0"});
  preprocParams.setParams(framePreprocessArg);

  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();

  ai_core::AlgoInput input = algoInput;

  ai_core::TensorData modelInput;
  preproc.process(input, preprocParams, modelInput);

  ai_core::TensorData modelOutput;
  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    engine.infer(modelInput, modelOutput);
  }
  // =================================================

  for (auto _ : state) {
    engine.infer(modelInput, modelOutput);
  }
}
BENCHMARK(BM_NCNN_CPU_DATA_YoloInfer)
    ->Repetitions(3)
    ->Iterations(30)
    ->Unit(benchmark::kMillisecond);
#endif

#ifdef WITH_TRT
static void BM_TRT_CPU_DATA_YoloInfer(benchmark::State &state) {
  ai_core::AlgoInferParams inferParams;
  inferParams.modelPath = "assets/models/yolov11n_trt_fp16.engine";
  inferParams.name = "yolov11n";
  inferParams.deviceType = ai_core::DeviceType::GPU;
  inferParams.dataType = ai_core::DataType::FLOAT32;

  ai_core::dnn::AlgoInferEngine engine("TrtAlgoInference", inferParams);
  engine.initialize();

  ai_core::AlgoPreprocParams preprocParams;
  ai_core::FramePreprocessArg framePreprocessArg = getFramePreprocessArg(
      ai_core::DataType::FLOAT32,
      ai_core::FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC,
      ai_core::BufferLocation::CPU, {"images"});
  preprocParams.setParams(framePreprocessArg);

  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();

  ai_core::AlgoInput input = algoInput;

  ai_core::TensorData modelInput;
  preproc.process(input, preprocParams, modelInput);

  ai_core::TensorData modelOutput;
  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    engine.infer(modelInput, modelOutput);
  }
  // =================================================

  for (auto _ : state) {
    engine.infer(modelInput, modelOutput);
  }
}

static void BM_TRT_GPU_DATA_YoloInfer(benchmark::State &state) {
  ai_core::AlgoInferParams inferParams;
  inferParams.modelPath = "assets/models/yolov11n_trt_fp16.engine";
  inferParams.name = "yolov11n";
  inferParams.deviceType = ai_core::DeviceType::GPU;
  inferParams.dataType = ai_core::DataType::FLOAT32;

  ai_core::dnn::AlgoInferEngine engine("TrtAlgoInference", inferParams);
  engine.initialize();

  ai_core::AlgoPreprocParams preprocParams;
  ai_core::FramePreprocessArg framePreprocessArg = getFramePreprocessArg(
      ai_core::DataType::FLOAT32,
      ai_core::FramePreprocessArg::FramePreprocType::CUDA_GPU_GENERIC,
      ai_core::BufferLocation::GPU_DEVICE, {"images"});
  preprocParams.setParams(framePreprocessArg);

  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();

  ai_core::AlgoInput input = algoInput;

  ai_core::TensorData modelInput;
  preproc.process(input, preprocParams, modelInput);

  ai_core::TensorData modelOutput;
  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    engine.infer(modelInput, modelOutput);
  }
  // =================================================

  for (auto _ : state) {
    engine.infer(modelInput, modelOutput);
  }
}
BENCHMARK(BM_TRT_CPU_DATA_YoloInfer)
    ->Repetitions(3)
    ->Iterations(30)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_TRT_GPU_DATA_YoloInfer)
    ->Repetitions(3)
    ->Iterations(30)
    ->Unit(benchmark::kMillisecond);
#endif
