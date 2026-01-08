#include "ai_core/algo_data_types.hpp"
#include "ai_core/algo_input_types.hpp"
#include "ai_core/infer_base.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/postproc_base.hpp"
#include "ai_core/preproc_base.hpp"
#include "postproc/anchor_det_postproc.hpp"
#include "preproc/frame_prep.hpp"
#include "gtest/gtest.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#ifdef WITH_TRT
#include "trt/trt_infer.hpp"

namespace testing_trt_infer {
namespace fs = std::filesystem;

using namespace ai_core;
using namespace ai_core::dnn;

class TrtInferenceTest : public ::testing::Test {
protected:
  void SetUp() override {
    ai_core::logging::Logger::instance().setLevel(
        ai_core::logging::LogLevel::Trace);
    ai_core::logging::Logger::instance().enableConsole(true);
    ai_core::logging::Logger::instance().enableFile(false);
    ai_core::logging::Logger::instance().enableColor(true);

    framePreproc = std::make_shared<FramePreprocess>();
    ASSERT_NE(framePreproc, nullptr);

    yoloDetPostproc = std::make_shared<AnchorDetPostproc>();
    ASSERT_NE(yoloDetPostproc, nullptr);
  }

  void CheckResults(const DetRet *detRet) {
    ASSERT_NE(detRet, nullptr);
    ASSERT_GE(detRet->bboxes.size(), 1);

    bool foundLabel0 = false;
    for (const auto &bbox : detRet->bboxes) {
      if (bbox.label == 0) {
        foundLabel0 = true;
        EXPECT_NEAR(bbox.score, 0.811, 0.05); // Slightly relaxed tolerance
        break;
      }
    }
    EXPECT_TRUE(foundLabel0) << "Expected to find detection with label 0";
  }

  FramePreprocessArg getPreprocParams() {
    FramePreprocessArg framePreprocessArg;
    framePreprocessArg.modelInputShape = {640, 640, 3};
    framePreprocessArg.dataType = DataType::FLOAT32;
    framePreprocessArg.needResize = true;
    framePreprocessArg.isEqualScale = true;
    framePreprocessArg.pad = {0, 0, 0};
    framePreprocessArg.meanVals = {0, 0, 0};
    framePreprocessArg.normVals = {255.f, 255.f, 255.f};
    framePreprocessArg.hwc2chw = true;
    framePreprocessArg.inputNames = {"images"};
    framePreprocessArg.preprocTaskType =
        FramePreprocessArg::FramePreprocType::CUDA_GPU_GENERIC;
    framePreprocessArg.outputLocation = BufferLocation::GPU_DEVICE;
    return framePreprocessArg;
  }

  AnchorDetParams getPostprocParams() {
    AnchorDetParams anchorDetParams;
    anchorDetParams.algoType = AnchorDetParams::AlgoType::YOLO_DET_V11;
    anchorDetParams.condThre = 0.5f;
    anchorDetParams.nmsThre = 0.45f;
    anchorDetParams.outputNames = {"output0"};
    return anchorDetParams;
  }

  // Prepare model input from image
  std::pair<TensorData, std::shared_ptr<RuntimeContext>>
  prepareInput(const cv::Mat &imageRGB) {
    AlgoPreprocParams preprocParams;
    preprocParams.setParams(getPreprocParams());
    AlgoInput algoInput;
    FrameInput frameInput;
    frameInput.image = std::make_shared<cv::Mat>(imageRGB);
    frameInput.inputRoi =
        std::make_shared<cv::Rect>(2, 2, imageRGB.cols - 4, imageRGB.rows - 4);
    algoInput.setParams(frameInput);

    std::shared_ptr<RuntimeContext> runtimeContext =
        std::make_shared<RuntimeContext>();
    TensorData modelInput;
    framePreproc->process(algoInput, preprocParams, modelInput, runtimeContext);

    return {modelInput, runtimeContext};
  }

  std::shared_ptr<IInferEnginePlugin> createEngine() {
    AlgoConstructParams tempInferParams;
    AlgoInferParams inferParams;
    inferParams.dataType = DataType::FLOAT32;
    inferParams.modelPath = "assets/models/yolov11n_trt_fp16.engine";
    inferParams.name = "yolov11n";
    inferParams.deviceType = DeviceType::GPU;
    inferParams.needDecrypt = false;
    tempInferParams.setParam("params", inferParams);

    return std::make_shared<TrtAlgoInference>(tempInferParams);
  }

  fs::path resourceDir = fs::path("assets");
  fs::path dataDir = resourceDir / "data";
  std::string imagePath = (dataDir / "yolov11/image.png").string();

  std::shared_ptr<IPreprocssPlugin> framePreproc;
  std::shared_ptr<IPostprocssPlugin> yoloDetPostproc;
};

TEST_F(TrtInferenceTest, AsyncCapabilityDetection) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  // Test dynamic_pointer_cast to IAsyncInferEngine
  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr)
      << "TrtAlgoInference should support IAsyncInferEngine";

  // Verify we can create a stream
  auto stream = asyncEngine->createStream();
  ASSERT_NE(stream, nullptr);

  engine->terminate();
}

// ============================================================================
// Test: Single stream async inference without CUDA Graph
// ============================================================================
TEST_F(TrtInferenceTest, SingleStreamAsyncWithoutGraph) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  // Create stream
  auto stream = asyncEngine->createStream();
  ASSERT_NE(stream, nullptr);

  // Ensure graph is disabled
  ASSERT_EQ(stream->setGraphEnabled(false), InferErrorCode::SUCCESS);
  ASSERT_FALSE(stream->isGraphEnabled());

  // Load and preprocess image
  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  auto [modelInput, runtimeContext] = prepareInput(imageRGB);

  TensorData modelOutput;

  // Async inference
  auto future = stream->inferAsync(modelInput, modelOutput);
  ASSERT_TRUE(future.valid());

  // Wait for completion
  auto status = future.get();
  ASSERT_EQ(status, InferErrorCode::SUCCESS);

  // Post-process and verify
  AlgoPostprocParams postprocParams;
  postprocParams.setParams(getPostprocParams());

  AlgoOutput algoOutput;
  ASSERT_TRUE(yoloDetPostproc->process(modelOutput, postprocParams, algoOutput,
                                       runtimeContext));

  auto *detRet = algoOutput.getParams<DetRet>();
  CheckResults(detRet);

  engine->terminate();
}

// ============================================================================
// Test: Single stream async inference with CUDA Graph
// ============================================================================
TEST_F(TrtInferenceTest, SingleStreamAsyncWithGraph) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  auto stream = asyncEngine->createStream();
  ASSERT_NE(stream, nullptr);

  // Enable CUDA Graph
  ASSERT_EQ(stream->setGraphEnabled(true), InferErrorCode::SUCCESS);
  ASSERT_TRUE(stream->isGraphEnabled());

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  auto [modelInput, runtimeContext] = prepareInput(imageRGB);

  // Run multiple iterations to test graph capture and replay
  const int numIterations = 5;
  for (int i = 0; i < numIterations; ++i) {
    TensorData modelOutput;
    auto future = stream->inferAsync(modelInput, modelOutput);
    auto status = future.get();
    ASSERT_EQ(status, InferErrorCode::SUCCESS) << "Failed at iteration " << i;

    // Verify results on first and last iteration
    if (i == 0 || i == numIterations - 1) {
      AlgoPostprocParams postprocParams;
      postprocParams.setParams(getPostprocParams());

      AlgoOutput algoOutput;
      ASSERT_TRUE(yoloDetPostproc->process(modelOutput, postprocParams,
                                           algoOutput, runtimeContext));
      auto *detRet = algoOutput.getParams<DetRet>();
      CheckResults(detRet);
    }
  }

  engine->terminate();
}

// ============================================================================
// Test: Stream pool creation and usage
// ============================================================================
TEST_F(TrtInferenceTest, StreamPoolUsage) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  // Create stream pool
  const size_t poolSize = 3;
  auto streamPool = asyncEngine->createStreamPool(poolSize);
  ASSERT_EQ(streamPool.size(), poolSize);

  for (const auto &stream : streamPool) {
    ASSERT_NE(stream, nullptr);
  }

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  // Pipeline-style execution
  const int numInferences = 10;
  std::vector<std::future<InferErrorCode>> futures;
  std::vector<TensorData> outputs(numInferences);

  for (int i = 0; i < numInferences; ++i) {
    auto &stream = streamPool[i % poolSize];
    auto [modelInput, runtimeContext] = prepareInput(imageRGB);

    // Wait for previous inference on this stream
    if (i >= static_cast<int>(poolSize) && futures[i - poolSize].valid()) {
      futures[i - poolSize].get();
    }

    futures.push_back(stream->inferAsync(modelInput, outputs[i]));
  }

  // Wait for remaining futures
  for (auto &f : futures) {
    if (f.valid()) {
      EXPECT_EQ(f.get(), InferErrorCode::SUCCESS);
    }
  }

  engine->terminate();
}

// ============================================================================
// Test: StreamContext with pre-allocated pinned buffers
// ============================================================================
TEST_F(TrtInferenceTest, StreamContextPreallocatedBuffers) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  // Create stream context with pre-allocated buffers
  auto ctx = asyncEngine->createStreamContext();
  ASSERT_NE(ctx.stream, nullptr);

  // Verify pre-allocated buffers exist
  const auto &modelInfo = engine->getModelInfo();
  for (const auto &input : modelInfo.inputs) {
    auto it = ctx.pinnedInputs.datas.find(input.name);
    EXPECT_NE(it, ctx.pinnedInputs.datas.end())
        << "Missing pre-allocated input buffer: " << input.name;
    if (it != ctx.pinnedInputs.datas.end()) {
      EXPECT_TRUE(it->second.isPinned())
          << "Input buffer should be pinned: " << input.name;
    }
  }

  for (const auto &output : modelInfo.outputs) {
    auto it = ctx.pinnedOutputs.datas.find(output.name);
    EXPECT_NE(it, ctx.pinnedOutputs.datas.end())
        << "Missing pre-allocated output buffer: " << output.name;
    if (it != ctx.pinnedOutputs.datas.end()) {
      EXPECT_TRUE(it->second.isPinned())
          << "Output buffer should be pinned: " << output.name;
    }
  }

  engine->terminate();
}

// ============================================================================
// Test: Allocate pinned host buffer
// ============================================================================
TEST_F(TrtInferenceTest, AllocatePinnedHostBuffer) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  // Allocate pinned buffer
  const size_t bufferSize = 1024 * 1024; // 1MB
  auto pinnedBuffer =
      asyncEngine->allocatePinnedHostBuffer(DataType::FLOAT32, bufferSize);

  EXPECT_EQ(pinnedBuffer.location(), BufferLocation::CPU);
  EXPECT_EQ(pinnedBuffer.memoryType(), BufferMemoryType::Pinned);
  EXPECT_TRUE(pinnedBuffer.isPinned());
  EXPECT_EQ(pinnedBuffer.getSizeBytes(), bufferSize);
  EXPECT_NE(pinnedBuffer.getRawHostPtr(), nullptr);

  engine->terminate();
}

// ============================================================================
// Test: Graph enable/disable toggle
// ============================================================================
TEST_F(TrtInferenceTest, GraphEnableDisableToggle) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  auto stream = asyncEngine->createStream();
  ASSERT_NE(stream, nullptr);

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  auto [modelInput, runtimeContext] = prepareInput(imageRGB);

  // Phase 1: Without graph
  stream->setGraphEnabled(false);
  ASSERT_FALSE(stream->isGraphEnabled());
  {
    TensorData output;
    auto future = stream->inferAsync(modelInput, output);
    ASSERT_EQ(future.get(), InferErrorCode::SUCCESS);
  }

  // Phase 2: Enable graph
  stream->setGraphEnabled(true);
  ASSERT_TRUE(stream->isGraphEnabled());
  for (int i = 0; i < 3; ++i) {
    TensorData output;
    auto future = stream->inferAsync(modelInput, output);
    ASSERT_EQ(future.get(), InferErrorCode::SUCCESS);
  }

  // Phase 3: Disable graph again
  stream->setGraphEnabled(false);
  ASSERT_FALSE(stream->isGraphEnabled());
  {
    TensorData output;
    auto future = stream->inferAsync(modelInput, output);
    ASSERT_EQ(future.get(), InferErrorCode::SUCCESS);
  }

  engine->terminate();
}

// ============================================================================
// Test: Synchronize and isComplete
// ============================================================================
TEST_F(TrtInferenceTest, SynchronizeAndIsComplete) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  auto stream = asyncEngine->createStream();
  ASSERT_NE(stream, nullptr);

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  auto [modelInput, runtimeContext] = prepareInput(imageRGB);

  TensorData modelOutput;

  // Start async inference
  auto future = stream->inferAsync(modelInput, modelOutput);

  // Wait using synchronize()
  ASSERT_EQ(stream->synchronize(), InferErrorCode::SUCCESS);

  // After synchronize, should be complete
  EXPECT_TRUE(stream->isComplete());

  // Future should also be ready
  EXPECT_EQ(future.get(), InferErrorCode::SUCCESS);

  engine->terminate();
}

// ============================================================================
// Test: GetHandle returns valid stream handle
// ============================================================================
TEST_F(TrtInferenceTest, GetStreamHandle) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  auto stream = asyncEngine->createStream();
  ASSERT_NE(stream, nullptr);

  auto handle = stream->getHandle();
  EXPECT_TRUE(static_cast<bool>(handle)) << "Handle should be valid";
  EXPECT_NE(handle.raw(), nullptr);

  engine->terminate();
}

// ============================================================================
// Test: Backward compatibility - infer() still works
// ============================================================================
TEST_F(TrtInferenceTest, BackwardCompatibilityInfer) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  auto [modelInput, runtimeContext] = prepareInput(imageRGB);
  TensorData modelOutput;

  // Use the old synchronous infer() method
  ASSERT_EQ(engine->infer(modelInput, modelOutput), InferErrorCode::SUCCESS);

  AlgoPostprocParams postprocParams;
  postprocParams.setParams(getPostprocParams());

  AlgoOutput algoOutput;
  ASSERT_TRUE(yoloDetPostproc->process(modelOutput, postprocParams, algoOutput,
                                       runtimeContext));

  auto *detRet = algoOutput.getParams<DetRet>();
  CheckResults(detRet);

  engine->terminate();
}

// ============================================================================
// Test: Stress test - many streams created and destroyed
// ============================================================================
TEST_F(TrtInferenceTest, StressTestManyStreams) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  const int numIterations = 20;
  for (int i = 0; i < numIterations; ++i) {
    // Create stream
    auto stream = asyncEngine->createStream();
    ASSERT_NE(stream, nullptr);

    // Run one inference
    auto [modelInput, runtimeContext] = prepareInput(imageRGB);
    TensorData modelOutput;

    auto future = stream->inferAsync(modelInput, modelOutput);
    EXPECT_EQ(future.get(), InferErrorCode::SUCCESS)
        << "Failed at iteration " << i;

    // Stream goes out of scope and is destroyed
  }

  engine->terminate();
}

// ============================================================================
// Test: Performance comparison - with and without CUDA Graph
// ============================================================================
TEST_F(TrtInferenceTest, PerformanceComparisonWithGraph) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto asyncEngine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(asyncEngine, nullptr);

  cv::Mat image = cv::imread(imagePath);
  ASSERT_FALSE(image.empty());
  cv::Mat imageRGB;
  cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

  auto [modelInput, runtimeContext] = prepareInput(imageRGB);

  const int warmupIterations = 5;
  const int benchIterations = 50;

  // Benchmark without graph
  {
    auto stream = asyncEngine->createStream();
    stream->setGraphEnabled(false);

    // Warmup
    for (int i = 0; i < warmupIterations; ++i) {
      TensorData output;
      stream->inferAsync(modelInput, output).get();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchIterations; ++i) {
      TensorData output;
      stream->inferAsync(modelInput, output).get();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double avgMs =
        std::chrono::duration<double, std::milli>(end - start).count() /
        benchIterations;
    LOG_INFO_S << "Without CUDA Graph: " << avgMs << " ms/inference";
  }

  // Benchmark with graph
  {
    auto stream = asyncEngine->createStream();
    stream->setGraphEnabled(true);

    // Warmup (includes graph capture)
    for (int i = 0; i < warmupIterations; ++i) {
      TensorData output;
      stream->inferAsync(modelInput, output).get();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchIterations; ++i) {
      TensorData output;
      stream->inferAsync(modelInput, output).get();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double avgMs =
        std::chrono::duration<double, std::milli>(end - start).count() /
        benchIterations;
    LOG_INFO_S << "With CUDA Graph: " << avgMs << " ms/inference";
  }

  engine->terminate();
}
} // namespace testing_trt_infer
#endif