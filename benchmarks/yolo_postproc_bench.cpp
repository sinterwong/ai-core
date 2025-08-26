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
#include "ai_core/algo_data_types.hpp"
#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/algo_postproc.hpp"
#include "ai_core/algo_preproc.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/tensor_data.hpp"
#include <benchmark/benchmark.h>

#include <memory>
#include <opencv2/opencv.hpp>

#ifdef WITH_ORT
const static auto engine = []() {
  ai_core::AlgoInferParams inferParams;
  inferParams.modelPath = "assets/models/yolov11n-fp16.onnx";
  inferParams.name = "yolov11n";
  inferParams.deviceType = ai_core::DeviceType::CPU;
  inferParams.dataType = ai_core::DataType::FLOAT16;

  std::shared_ptr<ai_core::dnn::AlgoInferEngine> engine;
  engine = std::make_shared<ai_core::dnn::AlgoInferEngine>("OrtAlgoInference",
                                                           inferParams);
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

static void BM_CPU_YoloDetPostproc(benchmark::State &state) {
  ai_core::dnn::AlgoPreproc preproc("FramePreprocess");
  preproc.initialize();

  ai_core::AlgoPreprocParams preprocParams;
  ai_core::FramePreprocessArg framePreprocessArg;
  framePreprocessArg.modelInputShape = {640, 640, 3};
  framePreprocessArg.dataType = ai_core::DataType::FLOAT16;
  framePreprocessArg.needResize = true;
  framePreprocessArg.isEqualScale = true;
  framePreprocessArg.pad = {0, 0, 0};
  framePreprocessArg.meanVals = {0, 0, 0};
  framePreprocessArg.normVals = {255.f, 255.f, 255.f};
  framePreprocessArg.hwc2chw = true;
  framePreprocessArg.inputNames = {"images"};
  framePreprocessArg.preprocTaskType =
      ai_core::FramePreprocessArg::FramePreprocType::OPENCV_CPU_GENERIC;
  framePreprocessArg.outputLocation = ai_core::BufferLocation::CPU;
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
  preproc.process(input, preprocParams, modelInput);

  ai_core::TensorData modelOutput;
  engine->infer(modelInput, modelOutput);

  ai_core::dnn::AlgoPostproc postproc("AnchorDetPostproc");
  postproc.initialize();

  ai_core::AlgoPostprocParams postprocParams;
  ai_core::AnchorDetParams anchorDetParams;
  anchorDetParams.algoType = ai_core::AnchorDetParams::AlgoType::YOLO_DET_V11;
  anchorDetParams.condThre = 0.5f;
  anchorDetParams.nmsThre = 0.45f;
  anchorDetParams.outputNames = {"output0"};
  postprocParams.setParams(anchorDetParams);

  ai_core::AlgoOutput algoOutput;

  // ==================== WARM-UP ====================
  for (int i = 0; i < 10; ++i) {
    postproc.process(modelOutput, preprocParams, algoOutput, postprocParams);
  }
  // =================================================

  for (auto _ : state) {
    postproc.process(modelOutput, preprocParams, algoOutput, postprocParams);
  }
}
BENCHMARK(BM_CPU_YoloDetPostproc)
    ->Repetitions(3)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);
