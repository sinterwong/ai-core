/**
 * @file trt_infer_bench.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2026-01-08
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifdef WITH_TRT
#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_async.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/tensor_data.hpp"
#include "trt/trt_infer.hpp"
#include <benchmark/benchmark.h>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace ai_core;
using namespace ai_core::dnn;

// ============================================================================
// Configuration Constants
// ============================================================================

namespace config {
constexpr int kWarmupIterations = 10;
constexpr const char *kModelPath = "assets/models/yolov11n_trt_fp16.engine";
constexpr const char *kModelName = "yolov11n";

// Input sizes to test (batch, channels, height, width)
const std::vector<std::vector<int64_t>> kInputShapes = {
    {1, 3, 640, 640},
    {1, 3, 320, 320},
    {1, 3, 1280, 1280},
};

// Pipeline depths to test
const std::vector<int> kPipelineDepths = {2, 3, 4, 6};

// Thread counts to test
const std::vector<int> kThreadCounts = {1, 2, 4, 8};
} // namespace config

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Calculate tensor size in bytes
 */
static size_t calculateSizeBytes(const std::vector<int64_t> &shape,
                                 DataType dtype) {
  size_t elements = 1;
  for (auto dim : shape)
    elements *= dim;

  size_t elementSize = 4; // Default float32
  switch (dtype) {
  case DataType::FLOAT16:
    elementSize = 2;
    break;
  case DataType::FLOAT32:
    elementSize = 4;
    break;
  case DataType::INT8:
    elementSize = 1;
    break;
  case DataType::INT32:
    elementSize = 4;
    break;
  default:
    elementSize = 4;
  }
  return elements * elementSize;
}

/**
 * @brief Create pageable (non-pinned) input tensor
 */
static TensorData createPageableInput(const std::vector<int64_t> &shape,
                                      DataType dtype) {
  TensorData data;
  size_t sizeBytes = calculateSizeBytes(shape, dtype);

  std::vector<uint8_t> buffer(sizeBytes);
  float *ptr = reinterpret_cast<float *>(buffer.data());
  size_t numElements = sizeBytes / sizeof(float);
  for (size_t i = 0; i < numElements; ++i) {
    ptr[i] = static_cast<float>(i % 255) / 255.0f;
  }

  data.datas["images"] = TypedBuffer::createFromCpu(dtype, std::move(buffer));
  data.shapes["images"] = std::vector<int>(shape.begin(), shape.end());
  return data;
}

/**
 * @brief Create pinned memory input tensor
 */
static TensorData createPinnedInput(IAsyncInferEngine *engine,
                                    const std::vector<int64_t> &shape,
                                    DataType dtype) {
  TensorData data;
  size_t sizeBytes = calculateSizeBytes(shape, dtype);

  auto buffer = engine->allocateAcceleratorBuffer(dtype, sizeBytes);

  float *ptr = buffer.getHostPtr<float>();
  size_t numElements = sizeBytes / sizeof(float);
  for (size_t i = 0; i < numElements; ++i) {
    ptr[i] = static_cast<float>(i % 255) / 255.0f;
  }

  data.datas["images"] = std::move(buffer);
  data.shapes["images"] = std::vector<int>(shape.begin(), shape.end());
  return data;
}

/**
 * @brief Perform warmup inference
 */
static void warmup(IInferEnginePlugin *engine, const TensorData &input,
                   int iterations = config::kWarmupIterations) {
  TensorData output;
  for (int i = 0; i < iterations; ++i) {
    engine->infer(input, output);
  }
}

static void warmupStream(IExecutionContext *stream, const TensorData &input,
                         int iterations = config::kWarmupIterations) {
  for (int i = 0; i < iterations; ++i) {
    TensorData output;
    stream->inferAsync(input, output).get();
  }
}

// ============================================================================
// Engine Manager: Thread-safe singleton for shared engine
// ============================================================================

class EngineManager {
public:
  static EngineManager &instance() {
    static EngineManager inst;
    return inst;
  }

  std::shared_ptr<TrtAlgoInference> getEngine() {
    std::call_once(initFlag_, [this]() { initializeEngine(); });
    return engine_;
  }

  std::shared_ptr<IAsyncInferEngine> getAsyncEngine() {
    return std::dynamic_pointer_cast<IAsyncInferEngine>(getEngine());
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (engine_) {
      engine_->terminate();
      engine_.reset();
    }
    // Reset the once_flag by recreating the manager
    // Note: In production, consider a more sophisticated approach
  }

private:
  EngineManager() = default;

  void initializeEngine() {
    AlgoConstructParams tempInferParams;
    AlgoInferParams inferParams;
    inferParams.modelPath = config::kModelPath;
    inferParams.name = config::kModelName;
    inferParams.deviceType = DeviceType::GPU;
    inferParams.dataType = DataType::FLOAT32;
    inferParams.needDecrypt = false;
    tempInferParams.setParam("params", inferParams);

    engine_ = std::make_shared<TrtAlgoInference>(tempInferParams);
    if (engine_->initialize() != InferErrorCode::SUCCESS) {
      throw std::runtime_error("Engine initialization failed");
    }
  }

  std::shared_ptr<TrtAlgoInference> engine_;
  std::once_flag initFlag_;
  std::mutex mutex_;
};

// ============================================================================
// Custom Counters for Rich Metrics
// ============================================================================

static void setCommonCounters(benchmark::State &state,
                              const std::vector<int64_t> &shape,
                              int itemsPerIteration = 1) {
  size_t inputBytes = calculateSizeBytes(shape, DataType::FLOAT32);
  // Estimate output bytes (YOLO output: 1 x 84 x 8400 for 640x640)
  size_t outputBytes = 84 * 8400 * sizeof(float);
  size_t totalBytes = inputBytes + outputBytes;

  state.SetItemsProcessed(state.iterations() * itemsPerIteration);
  state.SetBytesProcessed(state.iterations() * totalBytes * itemsPerIteration);

  // Custom counters
  state.counters["InputMB"] = inputBytes / (1024.0 * 1024.0);
  state.counters["Latency_us"] = benchmark::Counter(
      state.iterations(),
      benchmark::Counter::kIsRate | benchmark::Counter::kInvert,
      benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Baseline Comparisons (Single Thread, Fixed Shape)
// ============================================================================

/**
 * @brief Baseline: Legacy synchronous interface
 */
static void BM_TRT_Baseline_Sync(benchmark::State &state) {
  auto engine = EngineManager::instance().getEngine();
  auto input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);

  // Warmup (critical for fair comparison)
  warmup(engine.get(), input);

  TensorData output;
  for (auto _ : state) {
    auto result = engine->infer(input, output);
    if (result != InferErrorCode::SUCCESS) {
      state.SkipWithError("Inference failed");
      return;
    }
  }

  setCommonCounters(state, {1, 3, 640, 640});
}
BENCHMARK(BM_TRT_Baseline_Sync)->Unit(benchmark::kMillisecond)->Iterations(100);

/**
 * @brief Async without CUDA Graph (measures async overhead)
 */
static void BM_TRT_Async_NoGraph_Pageable(benchmark::State &state) {
  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto stream = asyncEngine->createExecutionContext();
  stream->setGraphEnabled(false);

  auto input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);
  warmupStream(stream.get(), input);

  for (auto _ : state) {
    TensorData output;
    auto future = stream->inferAsync(input, output);
    auto result = future.get();
    if (result != InferErrorCode::SUCCESS) {
      state.SkipWithError("Async inference failed");
      return;
    }
  }

  setCommonCounters(state, {1, 3, 640, 640});
}
BENCHMARK(BM_TRT_Async_NoGraph_Pageable)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

/**
 * @brief Async with CUDA Graph (measures graph benefit)
 */
static void BM_TRT_Async_WithGraph_Pageable(benchmark::State &state) {
  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto stream = asyncEngine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);
  warmupStream(stream.get(), input); // Graph captured during warmup

  for (auto _ : state) {
    TensorData output;
    auto future = stream->inferAsync(input, output);
    auto result = future.get();
    if (result != InferErrorCode::SUCCESS) {
      state.SkipWithError("Async inference failed");
      return;
    }
  }

  setCommonCounters(state, {1, 3, 640, 640});
}
BENCHMARK(BM_TRT_Async_WithGraph_Pageable)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

/**
 * @brief Async with Graph + Pinned Memory (best single-stream latency)
 */
static void BM_TRT_Async_WithGraph_Pinned(benchmark::State &state) {
  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto stream = asyncEngine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input =
      createPinnedInput(asyncEngine.get(), {1, 3, 640, 640}, DataType::FLOAT32);
  warmupStream(stream.get(), input);

  for (auto _ : state) {
    TensorData output;
    auto future = stream->inferAsync(input, output);
    auto result = future.get();
    if (result != InferErrorCode::SUCCESS) {
      state.SkipWithError("Async inference failed");
      return;
    }
  }

  setCommonCounters(state, {1, 3, 640, 640});
}
BENCHMARK(BM_TRT_Async_WithGraph_Pinned)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ============================================================================
// Memory Type Impact Analysis
// ============================================================================

/**
 * @brief Compare Pageable vs Pinned memory transfer overhead
 * Arg: 0 = Pageable, 1 = Pinned
 */
static void BM_TRT_MemoryType_Comparison(benchmark::State &state) {
  const bool usePinned = state.range(0) == 1;

  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto stream = asyncEngine->createExecutionContext();
  stream->setGraphEnabled(false); // Disable graph to isolate memory impact

  TensorData input;
  if (usePinned) {
    input = createPinnedInput(asyncEngine.get(), {1, 3, 640, 640},
                              DataType::FLOAT32);
  } else {
    input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);
  }

  warmupStream(stream.get(), input);

  for (auto _ : state) {
    TensorData output;
    stream->inferAsync(input, output).get();
  }

  state.SetLabel(usePinned ? "Pinned" : "Pageable");
  setCommonCounters(state, {1, 3, 640, 640});
}
BENCHMARK(BM_TRT_MemoryType_Comparison)
    ->Arg(0) // Pageable
    ->Arg(1) // Pinned
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ============================================================================
// Input Size Scaling
// ============================================================================

// ============================================================================
// Pipeline Throughput Analysis
// ============================================================================

/**
 * @brief Measure throughput with different pipeline depths
 * Tests latency hiding effectiveness
 */
static void BM_TRT_Pipeline_Throughput(benchmark::State &state) {
  const int pipelineDepth = state.range(0);

  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto streamPool = asyncEngine->createContextPool(pipelineDepth);

  // Enable graph for all streams
  for (auto &s : streamPool) {
    s->setGraphEnabled(true);
  }

  auto input =
      createPinnedInput(asyncEngine.get(), {1, 3, 640, 640}, DataType::FLOAT32);

  // Warmup all streams
  for (auto &s : streamPool) {
    warmupStream(s.get(), input, 5);
  }

  // Pre-allocate output containers
  std::vector<TensorData> outputs(pipelineDepth);

  for (auto _ : state) {
    std::vector<std::future<InferErrorCode>> futures;
    futures.reserve(pipelineDepth);

    // Submit all tasks (non-blocking)
    for (int i = 0; i < pipelineDepth; ++i) {
      futures.push_back(streamPool[i]->inferAsync(input, outputs[i]));
    }

    // Wait for all completions
    for (auto &f : futures) {
      if (f.get() != InferErrorCode::SUCCESS) {
        state.SkipWithError("Pipeline inference failed");
        return;
      }
    }
  }

  state.SetLabel("Depth=" + std::to_string(pipelineDepth));
  state.SetItemsProcessed(state.iterations() * pipelineDepth);
  state.counters["Throughput"] = benchmark::Counter(
      state.iterations() * pipelineDepth, benchmark::Counter::kIsRate,
      benchmark::Counter::OneK::kIs1000);
}
BENCHMARK(BM_TRT_Pipeline_Throughput)
    ->Arg(2)
    ->Arg(3)
    ->Arg(4)
    ->Arg(6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50);

// ============================================================================
// CUDA Graph Overhead Analysis
// ============================================================================

/**
 * @brief Measure CUDA Graph capture overhead (one-time cost)
 */
static void BM_TRT_Graph_Capture_Overhead(benchmark::State &state) {
  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto input =
      createPinnedInput(asyncEngine.get(), {1, 3, 640, 640}, DataType::FLOAT32);

  for (auto _ : state) {
    // Create fresh stream for each iteration
    auto stream = asyncEngine->createExecutionContext();
    stream->setGraphEnabled(true);

    // First inference captures the graph
    TensorData output;
    auto result = stream->inferAsync(input, output).get();
    if (result != InferErrorCode::SUCCESS) {
      state.SkipWithError("Graph capture failed");
      return;
    }
  }

  state.SetLabel("Graph Capture");
}
BENCHMARK(BM_TRT_Graph_Capture_Overhead)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20);

/**
 * @brief Measure CUDA Graph replay latency (amortized benefit)
 */
static void BM_TRT_Graph_Replay_Latency(benchmark::State &state) {
  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto stream = asyncEngine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input =
      createPinnedInput(asyncEngine.get(), {1, 3, 640, 640}, DataType::FLOAT32);

  // Capture graph during warmup
  warmupStream(stream.get(), input);

  for (auto _ : state) {
    TensorData output;
    stream->inferAsync(input, output).get();
  }

  state.SetLabel("Graph Replay");
  setCommonCounters(state, {1, 3, 640, 640});
}
BENCHMARK(BM_TRT_Graph_Replay_Latency)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ============================================================================
// Graph Toggle Overhead (Shape Change Simulation)
// ============================================================================

/**
 * @brief Measure overhead when Graph needs re-capture (shape change)
 * This simulates dynamic shape scenarios
 */
static void BM_TRT_Graph_Recapture_Overhead(benchmark::State &state) {
  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto stream = asyncEngine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input640 =
      createPinnedInput(asyncEngine.get(), {1, 3, 640, 640}, DataType::FLOAT32);
  auto input320 =
      createPinnedInput(asyncEngine.get(), {1, 3, 320, 320}, DataType::FLOAT32);

  // Initial capture
  warmupStream(stream.get(), input640);

  bool toggle = false;
  for (auto _ : state) {
    // Alternate between two shapes (forces recapture)
    const auto &input = toggle ? input320 : input640;
    toggle = !toggle;

    TensorData output;
    stream->inferAsync(input, output).get();
  }

  state.SetLabel("Alternating Shapes");
}
BENCHMARK(BM_TRT_Graph_Recapture_Overhead)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(50);

// ============================================================================
// Stream Creation/Destruction Overhead
// ============================================================================

/**
 * @brief Measure stream creation overhead
 */
static void BM_TRT_Stream_Creation_Overhead(benchmark::State &state) {
  auto asyncEngine = EngineManager::instance().getAsyncEngine();

  for (auto _ : state) {
    auto stream = asyncEngine->createExecutionContext();
    benchmark::DoNotOptimize(stream);
  }

  state.SetLabel("Stream Create");
}
BENCHMARK(BM_TRT_Stream_Creation_Overhead)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

/**
 * @brief Measure stream creation + first inference (cold start)
 */
static void BM_TRT_Stream_ColdStart_Latency(benchmark::State &state) {
  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto input =
      createPinnedInput(asyncEngine.get(), {1, 3, 640, 640}, DataType::FLOAT32);

  for (auto _ : state) {
    auto stream = asyncEngine->createExecutionContext();
    stream->setGraphEnabled(false);

    TensorData output;
    stream->inferAsync(input, output).get();
  }

  state.SetLabel("Cold Start");
}
BENCHMARK(BM_TRT_Stream_ColdStart_Latency)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(20);

// ============================================================================
// Comprehensive Summary Comparison
// ============================================================================

/**
 * @brief All-in-one comparison for executive summary
 * Args: [GraphEnabled, PinnedMemory]
 */
static void BM_TRT_Summary_Comparison(benchmark::State &state) {
  const bool graphEnabled = state.range(0) == 1;
  const bool pinnedMemory = state.range(1) == 1;

  auto asyncEngine = EngineManager::instance().getAsyncEngine();
  auto stream = asyncEngine->createExecutionContext();
  stream->setGraphEnabled(graphEnabled);

  TensorData input;
  if (pinnedMemory) {
    input = createPinnedInput(asyncEngine.get(), {1, 3, 640, 640},
                              DataType::FLOAT32);
  } else {
    input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);
  }

  warmupStream(stream.get(), input);

  for (auto _ : state) {
    TensorData output;
    stream->inferAsync(input, output).get();
  }

  std::string label = std::string(graphEnabled ? "Graph" : "NoGraph") + "_" +
                      (pinnedMemory ? "Pinned" : "Pageable");
  state.SetLabel(label);
  setCommonCounters(state, {1, 3, 640, 640});
}
BENCHMARK(BM_TRT_Summary_Comparison)
    ->Args({0, 0}) // NoGraph, Pageable
    ->Args({0, 1}) // NoGraph, Pinned
    ->Args({1, 0}) // Graph, Pageable
    ->Args({1, 1}) // Graph, Pinned
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

#endif // WITH_TRT
