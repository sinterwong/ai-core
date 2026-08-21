#include "ai_core/algo_types.hpp"
#include "ai_core/i_infer_engine.hpp"
#include "ai_core/i_postprocess.hpp"
#include "ai_core/i_preprocess.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/input_types.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/opencv_interop.hpp"
#include "postproc/yolo_det.hpp"
#include "preproc/cuda_generic_preprocess.hpp"
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

    m_framePreproc = std::make_shared<CudaGenericPreprocess>();
    ASSERT_NE(m_framePreproc, nullptr);

    m_yoloDetPostproc = std::make_shared<Yolov11Det>();
    ASSERT_NE(m_yoloDetPostproc, nullptr);
  }

  void checkResults(const DetRet *det_ret) {
    ASSERT_NE(det_ret, nullptr);
    ASSERT_GE(det_ret->bboxes.size(), 1);

    bool found_label0 = false;
    for (const auto &bbox : det_ret->bboxes) {
      if (bbox.label == 0) {
        found_label0 = true;
        EXPECT_NEAR(bbox.score, 0.811, 0.05); // Slightly relaxed tolerance
        break;
      }
    }
    EXPECT_TRUE(found_label0) << "Expected to find detection with label 0";
  }

  FramePreprocessArg getPreprocParams() {
    FramePreprocessArg frame_preprocess_arg;
    frame_preprocess_arg.model_input_shape = {640, 640, 3};
    frame_preprocess_arg.data_type = DataType::FLOAT32;
    frame_preprocess_arg.need_resize = true;
    frame_preprocess_arg.is_equal_scale = true;
    frame_preprocess_arg.pad = {0, 0, 0};
    frame_preprocess_arg.mean_vals = {0, 0, 0};
    frame_preprocess_arg.std_vals = {255.f, 255.f, 255.f};
    frame_preprocess_arg.model_input_format = ai_core::ImagePixelFormat::RGB888;
    frame_preprocess_arg.hwc2chw = true;
    frame_preprocess_arg.input_names = {"images"};
    frame_preprocess_arg.output_location = BufferLocation::GpuDevice;
    return frame_preprocess_arg;
  }

  AnchorDetParams getPostprocParams() {
    AnchorDetParams anchor_det_params;
    anchor_det_params.cond_thre = 0.5f;
    anchor_det_params.nms_thre = 0.45f;
    anchor_det_params.output_names = {"output0"};
    return anchor_det_params;
  }

  std::pair<TensorData, std::shared_ptr<RuntimeContext>>
  prepareInput(const cv::Mat &image) {
    AlgoPreprocParams preproc_params;
    preproc_params.setParams(getPreprocParams());
    AlgoInput algo_input;
    FrameInput frame_input;
    frame_input.image = ai_core::interop::viewFromMat(image);
    frame_input.roi = ai_core::Rect{2, 2, image.cols - 4, image.rows - 4};
    algo_input.setParams(frame_input);

    std::shared_ptr<RuntimeContext> runtime_context =
        std::make_shared<RuntimeContext>();
    TensorData model_input;
    m_framePreproc->process(algo_input, preproc_params, model_input,
                            runtime_context);

    return {model_input, runtime_context};
  }

  std::shared_ptr<IInferEnginePlugin> createEngine() {
    AlgoConstructParams temp_infer_params;
    AlgoInferParams infer_params;
    infer_params.data_type = DataType::FLOAT32;
    infer_params.model_path = "assets/models/yolov11n_trt_fp16.engine";
    infer_params.name = "yolov11n";
    infer_params.device_type = DeviceType::GPU;
    infer_params.need_decrypt = false;
    temp_infer_params.setParam("params", infer_params);

    return std::make_shared<TrtAlgoInference>(temp_infer_params);
  }

  fs::path m_resourceDir = fs::path("assets");
  fs::path m_dataDir = m_resourceDir / "data";
  std::string m_image_path = (m_dataDir / "yolov11/image.png").string();

  std::shared_ptr<IPreprocessPlugin> m_framePreproc;
  std::shared_ptr<IPostprocessPlugin> m_yoloDetPostproc;
};

TEST_F(TrtInferenceTest, AsyncCapabilityDetection) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr)
      << "TrtAlgoInference should support IAsyncInferEngine";

  auto stream = async_engine->createExecutionContext();
  ASSERT_NE(stream, nullptr);

  engine->terminate();
}

TEST_F(TrtInferenceTest, SingleStreamAsyncWithoutGraph) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  auto stream = async_engine->createExecutionContext();
  ASSERT_NE(stream, nullptr);

  ASSERT_EQ(stream->setGraphEnabled(false), InferErrorCode::SUCCESS);
  ASSERT_FALSE(stream->isGraphEnabled());

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  auto [modelInput, runtime_context] = prepareInput(image);

  TensorData model_output;

  auto future = stream->inferAsync(modelInput, model_output);
  ASSERT_TRUE(future.valid());

  auto status = future.get();
  ASSERT_EQ(status, InferErrorCode::SUCCESS);

  AlgoPostprocParams postproc_params;
  postproc_params.setParams(getPostprocParams());

  AlgoOutput algo_output;
  ASSERT_EQ(m_yoloDetPostproc->process(model_output, postproc_params,
                                       algo_output, runtime_context),
            InferErrorCode::SUCCESS);

  auto *det_ret = algo_output.getParams<DetRet>();
  checkResults(det_ret);

  engine->terminate();
}

TEST_F(TrtInferenceTest, SingleStreamAsyncWithGraph) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  auto stream = async_engine->createExecutionContext();
  ASSERT_NE(stream, nullptr);

  ASSERT_EQ(stream->setGraphEnabled(true), InferErrorCode::SUCCESS);
  ASSERT_TRUE(stream->isGraphEnabled());

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  auto [modelInput, runtime_context] = prepareInput(image);

  const int num_iterations = 5;
  for (int i = 0; i < num_iterations; ++i) {
    TensorData model_output;
    auto future = stream->inferAsync(modelInput, model_output);
    auto status = future.get();
    ASSERT_EQ(status, InferErrorCode::SUCCESS) << "Failed at iteration " << i;

    if (i == 0 || i == num_iterations - 1) {
      AlgoPostprocParams postproc_params;
      postproc_params.setParams(getPostprocParams());

      AlgoOutput algo_output;
      ASSERT_EQ(m_yoloDetPostproc->process(model_output, postproc_params,
                                           algo_output, runtime_context),
                InferErrorCode::SUCCESS);
      auto *det_ret = algo_output.getParams<DetRet>();
      checkResults(det_ret);
    }
  }

  engine->terminate();
}

TEST_F(TrtInferenceTest, StreamPoolUsage) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  const size_t pool_size = 3;
  auto stream_pool = async_engine->createContextPool(pool_size);
  ASSERT_EQ(stream_pool.size(), pool_size);

  for (const auto &stream : stream_pool) {
    ASSERT_NE(stream, nullptr);
  }

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  const int num_inferences = 10;
  std::vector<std::future<InferErrorCode>> futures;
  std::vector<TensorData> outputs(num_inferences);

  for (int i = 0; i < num_inferences; ++i) {
    auto &stream = stream_pool[i % pool_size];
    auto [modelInput, runtime_context] = prepareInput(image);

    // A context cannot accept new work until its prior submission completes.
    if (i >= static_cast<int>(pool_size) && futures[i - pool_size].valid()) {
      futures[i - pool_size].get();
    }

    futures.push_back(stream->inferAsync(modelInput, outputs[i]));
  }

  for (auto &f : futures) {
    if (f.valid()) {
      EXPECT_EQ(f.get(), InferErrorCode::SUCCESS);
    }
  }

  engine->terminate();
}

TEST_F(TrtInferenceTest, StreamContextPreallocatedBuffers) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  auto ctx = async_engine->createContextPackage();
  ASSERT_NE(ctx.context, nullptr);

  const auto &model_info = engine->getModelInfo();
  for (const auto &input : model_info.inputs) {
    const ai_core::Tensor *t = ctx.inputs.find(input.name);
    ASSERT_NE(t, nullptr) << "Missing pre-allocated input buffer: "
                          << input.name;
    EXPECT_TRUE(t->buffer.isPinned())
        << "Input buffer should be pinned: " << input.name;
  }

  for (const auto &output : model_info.outputs) {
    const ai_core::Tensor *t = ctx.outputs.find(output.name);
    ASSERT_NE(t, nullptr) << "Missing pre-allocated output buffer: "
                          << output.name;
    EXPECT_TRUE(t->buffer.isPinned())
        << "Output buffer should be pinned: " << output.name;
  }

  engine->terminate();
}

TEST_F(TrtInferenceTest, allocateAcceleratorBuffer) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  const size_t buffer_size = 1024 * 1024;
  auto pinned_buffer =
      async_engine->allocateAcceleratorBuffer(DataType::FLOAT32, buffer_size);

  EXPECT_EQ(pinned_buffer.location(), BufferLocation::CPU);
  EXPECT_EQ(pinned_buffer.memoryType(), BufferMemoryType::Pinned);
  EXPECT_TRUE(pinned_buffer.isPinned());
  EXPECT_EQ(pinned_buffer.getSizeBytes(), buffer_size);
  EXPECT_NE(pinned_buffer.getRawHostPtr(), nullptr);

  engine->terminate();
}

TEST_F(TrtInferenceTest, GraphEnableDisableToggle) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  auto stream = async_engine->createExecutionContext();
  ASSERT_NE(stream, nullptr);

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  auto [modelInput, runtime_context] = prepareInput(image);

  stream->setGraphEnabled(false);
  ASSERT_FALSE(stream->isGraphEnabled());
  {
    TensorData output;
    auto future = stream->inferAsync(modelInput, output);
    ASSERT_EQ(future.get(), InferErrorCode::SUCCESS);
  }

  stream->setGraphEnabled(true);
  ASSERT_TRUE(stream->isGraphEnabled());
  for (int i = 0; i < 3; ++i) {
    TensorData output;
    auto future = stream->inferAsync(modelInput, output);
    ASSERT_EQ(future.get(), InferErrorCode::SUCCESS);
  }

  stream->setGraphEnabled(false);
  ASSERT_FALSE(stream->isGraphEnabled());
  {
    TensorData output;
    auto future = stream->inferAsync(modelInput, output);
    ASSERT_EQ(future.get(), InferErrorCode::SUCCESS);
  }

  engine->terminate();
}

TEST_F(TrtInferenceTest, SynchronizeAndIsComplete) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  auto stream = async_engine->createExecutionContext();
  ASSERT_NE(stream, nullptr);

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  auto [modelInput, runtime_context] = prepareInput(image);

  TensorData model_output;

  auto future = stream->inferAsync(modelInput, model_output);

  ASSERT_EQ(stream->synchronize(), InferErrorCode::SUCCESS);

  EXPECT_TRUE(stream->isComplete());

  EXPECT_EQ(future.get(), InferErrorCode::SUCCESS);

  engine->terminate();
}

TEST_F(TrtInferenceTest, GetStreamHandle) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  auto stream = async_engine->createExecutionContext();
  ASSERT_NE(stream, nullptr);

  auto handle = stream->getHandle();
  EXPECT_TRUE(static_cast<bool>(handle)) << "Handle should be valid";
  EXPECT_NE(handle.raw(), nullptr);

  engine->terminate();
}

TEST_F(TrtInferenceTest, BackwardCompatibilityInfer) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  auto [modelInput, runtime_context] = prepareInput(image);
  TensorData model_output;

  ASSERT_EQ(engine->infer(modelInput, model_output), InferErrorCode::SUCCESS);

  AlgoPostprocParams postproc_params;
  postproc_params.setParams(getPostprocParams());

  AlgoOutput algo_output;
  ASSERT_EQ(m_yoloDetPostproc->process(model_output, postproc_params,
                                       algo_output, runtime_context),
            InferErrorCode::SUCCESS);

  auto *det_ret = algo_output.getParams<DetRet>();
  checkResults(det_ret);

  engine->terminate();
}

TEST_F(TrtInferenceTest, StressTestManyStreams) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  const int num_iterations = 20;
  for (int i = 0; i < num_iterations; ++i) {
    auto stream = async_engine->createExecutionContext();
    ASSERT_NE(stream, nullptr);

    auto [modelInput, runtime_context] = prepareInput(image);
    TensorData model_output;

    auto future = stream->inferAsync(modelInput, model_output);
    EXPECT_EQ(future.get(), InferErrorCode::SUCCESS)
        << "Failed at iteration " << i;
  }

  engine->terminate();
}

TEST_F(TrtInferenceTest, PerformanceComparisonWithGraph) {
  auto engine = createEngine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(), InferErrorCode::SUCCESS);

  auto async_engine = std::dynamic_pointer_cast<IAsyncInferEngine>(engine);
  ASSERT_NE(async_engine, nullptr);

  cv::Mat image = cv::imread(m_image_path);
  ASSERT_FALSE(image.empty());

  auto [modelInput, runtime_context] = prepareInput(image);

  const int warmup_iterations = 5;
  const int bench_iterations = 50;

  // Benchmark without graph
  {
    auto stream = async_engine->createExecutionContext();
    stream->setGraphEnabled(false);

    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
      TensorData output;
      stream->inferAsync(modelInput, output).get();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iterations; ++i) {
      TensorData output;
      stream->inferAsync(modelInput, output).get();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double avg_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        bench_iterations;
    LOG_INFO_S << "Without CUDA Graph: " << avg_ms << " ms/inference";
  }

  // Benchmark with graph
  {
    auto stream = async_engine->createExecutionContext();
    stream->setGraphEnabled(true);

    // Warmup (includes graph capture)
    for (int i = 0; i < warmup_iterations; ++i) {
      TensorData output;
      stream->inferAsync(modelInput, output).get();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iterations; ++i) {
      TensorData output;
      stream->inferAsync(modelInput, output).get();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double avg_ms =
        std::chrono::duration<double, std::milli>(end - start).count() /
        bench_iterations;
    LOG_INFO_S << "With CUDA Graph: " << avg_ms << " ms/inference";
  }

  engine->terminate();
}
} // namespace testing_trt_infer
#endif
