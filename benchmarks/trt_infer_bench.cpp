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
#include "ai_core/algo_types.hpp"
#include "ai_core/infer_async.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/tensor_data.hpp"
#include "trt/trt_infer.hpp"
#include <atomic>
#include <benchmark/benchmark.h>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

using namespace ai_core;
using namespace ai_core::dnn;

// ============================================================================
// Configuration Constants
// ============================================================================

namespace config {
constexpr int k_warmup_iterations = 10;
constexpr const char *k_model_path = "assets/models/yolov11n_trt_fp16.engine";
constexpr const char *k_model_name = "yolov11n";

// Input sizes to test (batch, channels, height, width)
const std::vector<std::vector<int64_t>> k_input_shapes = {
    {1, 3, 640, 640},
    {1, 3, 320, 320},
    {1, 3, 1280, 1280},
};

// Pipeline depths to test
const std::vector<int> k_pipeline_depths = {2, 3, 4, 6};

// Thread counts to test
const std::vector<int> k_thread_counts = {1, 2, 4, 8};
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

  size_t element_size = 4; // Default float32
  switch (dtype) {
  case DataType::FLOAT16:
    element_size = 2;
    break;
  case DataType::FLOAT32:
    element_size = 4;
    break;
  case DataType::INT8:
    element_size = 1;
    break;
  case DataType::INT32:
    element_size = 4;
    break;
  default:
    element_size = 4;
  }
  return elements * element_size;
}

/**
 * @brief Create pageable (non-pinned) input tensor
 */
static TensorData createPageableInput(const std::vector<int64_t> &shape,
                                      DataType dtype) {
  TensorData data;
  size_t size_bytes = calculateSizeBytes(shape, dtype);

  std::vector<uint8_t> buffer(size_bytes);
  float *ptr = reinterpret_cast<float *>(buffer.data());
  size_t num_elements = size_bytes / sizeof(float);
  for (size_t i = 0; i < num_elements; ++i) {
    ptr[i] = static_cast<float>(i % 255) / 255.0f;
  }

  data.set("images", TypedBuffer::createFromCpu(dtype, std::move(buffer)),
           std::vector<int>(shape.begin(), shape.end()));
  return data;
}

/**
 * @brief Create pinned memory input tensor
 */
static TensorData createPinnedInput(IAsyncInferEngine *engine,
                                    const std::vector<int64_t> &shape,
                                    DataType dtype) {
  TensorData data;
  size_t size_bytes = calculateSizeBytes(shape, dtype);

  auto buffer = engine->allocateAcceleratorBuffer(dtype, size_bytes);

  float *ptr = buffer.getHostPtr<float>();
  size_t num_elements = size_bytes / sizeof(float);
  for (size_t i = 0; i < num_elements; ++i) {
    ptr[i] = static_cast<float>(i % 255) / 255.0f;
  }

  data.set("images", std::move(buffer),
           std::vector<int>(shape.begin(), shape.end()));
  return data;
}

/**
 * @brief Perform warmup inference
 */
static void warmup(IInferEnginePlugin *engine, const TensorData &input,
                   int iterations = config::k_warmup_iterations) {
  TensorData output;
  for (int i = 0; i < iterations; ++i) {
    engine->infer(input, output);
  }
}

static void warmupStream(IExecutionContext *stream, const TensorData &input,
                         int iterations = config::k_warmup_iterations) {
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
    // static EngineManager inst;
    // return inst;

    // 使用不朽单例
    static EngineManager *inst = new EngineManager();
    return *inst;
  }

  std::shared_ptr<TrtAlgoInference> getEngine() {
    std::call_once(m_initFlag, [this]() { initializeEngine(); });
    return m_engine;
  }

  std::shared_ptr<IAsyncInferEngine> getAsyncEngine() {
    return std::dynamic_pointer_cast<IAsyncInferEngine>(getEngine());
  }

  void reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_engine) {
      m_engine->terminate();
      m_engine.reset();
    }
    // Reset the once_flag by recreating the manager
    // Note: In production, consider a more sophisticated approach
  }

private:
  EngineManager() = default;

  void initializeEngine() {
    AlgoConstructParams temp_infer_params;
    AlgoInferParams infer_params;
    infer_params.model_path = config::k_model_path;
    infer_params.name = config::k_model_name;
    infer_params.device_type = DeviceType::GPU;
    infer_params.data_type = DataType::FLOAT32;
    infer_params.need_decrypt = false;
    temp_infer_params.setParam("params", infer_params);

    m_engine = std::make_shared<TrtAlgoInference>(temp_infer_params);
    if (m_engine->initialize() != InferErrorCode::SUCCESS) {
      throw std::runtime_error("Engine initialization failed");
    }
  }

  std::shared_ptr<TrtAlgoInference> m_engine;
  std::once_flag m_initFlag;
  std::mutex m_mutex;
};

// ============================================================================
// Custom Counters for Rich Metrics
// ============================================================================

static void setCommonCounters(benchmark::State &state,
                              const std::vector<int64_t> &shape,
                              int items_per_iteration = 1) {
  size_t input_bytes = calculateSizeBytes(shape, DataType::FLOAT32);
  // Estimate output bytes (YOLO output: 1 x 84 x 8400 for 640x640)
  size_t output_bytes = 84 * 8400 * sizeof(float);
  size_t total_bytes = input_bytes + output_bytes;

  state.SetItemsProcessed(state.iterations() * items_per_iteration);
  state.SetBytesProcessed(state.iterations() * total_bytes *
                          items_per_iteration);

  // Custom counters
  state.counters["InputMB"] = input_bytes / (1024.0 * 1024.0);
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
 * @brief Concurrent synchronous infer() throughput vs thread count.
 *
 * google-benchmark runs the loop body on `threads` threads simultaneously
 * against the shared engine. Because sync infer() now borrows an execution
 * context from a pool (no global mutex), items/s should scale with threads.
 * Acceptance (v1.7): items/s at 4 threads >= 3x the 1-thread rate.
 */
static void BM_TRT_Sync_Concurrent(benchmark::State &state) {
  auto engine = EngineManager::instance().getEngine();
  // Each thread owns its input/output so nothing but the engine is shared.
  auto input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);
  if (state.thread_index() == 0) {
    warmup(engine.get(), input);
  }

  TensorData output;
  for (auto _ : state) {
    auto result = engine->infer(input, output);
    if (result != InferErrorCode::SUCCESS) {
      state.SkipWithError("Inference failed");
      return;
    }
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TRT_Sync_Concurrent)
    ->ThreadRange(1, 8)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

/**
 * @brief Unambiguous aggregate-throughput sweep for concurrent sync infer().
 *
 * Spawns N worker threads that each call the shared engine's infer() in a
 * tight loop for a fixed wall-clock window, then reports the aggregate
 * images/sec at N = 1, 2, 4, 8. This measures the real scaling the context
 * pool provides (google-benchmark's Threads() aggregation is easy to
 * misread). Registered as a single benchmark that runs the whole sweep once.
 */
static void BM_TRT_Sync_ThroughputSweep(benchmark::State &state) {
  auto engine = EngineManager::instance().getEngine();
  auto warmup_input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);
  warmup(engine.get(), warmup_input, 20);

  const std::vector<int> thread_counts = {1, 2, 4, 8};
  const auto window = std::chrono::milliseconds(1500);

  double single_thread_rate = 0.0;
  for (auto _ : state) {
    for (int n : thread_counts) {
      std::atomic<uint64_t> total_ops{0};
      std::atomic<bool> go{false};
      std::atomic<bool> stop{false};
      std::vector<std::thread> workers;
      workers.reserve(n);
      auto async_engine = EngineManager::instance().getAsyncEngine();
      for (int t = 0; t < n; ++t) {
        workers.emplace_back([&]() {
          // Pinned input so per-stream H2D copies are truly async and can
          // overlap across threads (pageable copies serialize on the driver).
          auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                         DataType::FLOAT32);
          TensorData output;
          while (!go.load(std::memory_order_acquire)) {
          }
          uint64_t local = 0;
          while (!stop.load(std::memory_order_acquire)) {
            if (engine->infer(input, output) == InferErrorCode::SUCCESS) {
              ++local;
            }
          }
          total_ops.fetch_add(local, std::memory_order_relaxed);
        });
      }
      auto start = std::chrono::steady_clock::now();
      go.store(true, std::memory_order_release);
      std::this_thread::sleep_for(window);
      stop.store(true, std::memory_order_release);
      for (auto &w : workers) {
        w.join();
      }
      auto elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
      double rate = total_ops.load() / elapsed;
      if (n == 1) {
        single_thread_rate = rate;
      }
      state.counters["thr" + std::to_string(n) + "_imgps"] = rate;
      if (single_thread_rate > 0) {
        state.counters["thr" + std::to_string(n) + "_speedup"] =
            rate / single_thread_rate;
      }
    }
  }
}
BENCHMARK(BM_TRT_Sync_ThroughputSweep)
    ->Iterations(1)
    ->Unit(benchmark::kMillisecond);

/**
 * @brief Async without CUDA Graph (measures async overhead)
 */
static void BM_TRT_Async_NoGraph_Pageable(benchmark::State &state) {
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
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
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
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
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                 DataType::FLOAT32);
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
  const bool use_pinned = state.range(0) == 1;

  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  stream->setGraphEnabled(false); // Disable graph to isolate memory impact

  TensorData input;
  if (use_pinned) {
    input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                              DataType::FLOAT32);
  } else {
    input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);
  }

  warmupStream(stream.get(), input);

  for (auto _ : state) {
    TensorData output;
    stream->inferAsync(input, output).get();
  }

  state.SetLabel(use_pinned ? "Pinned" : "Pageable");
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
  const int pipeline_depth = state.range(0);

  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream_pool = async_engine->createContextPool(pipeline_depth);

  // Enable graph for all streams
  for (auto &s : stream_pool) {
    s->setGraphEnabled(true);
  }

  auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                 DataType::FLOAT32);

  // Warmup all streams
  for (auto &s : stream_pool) {
    warmupStream(s.get(), input, 5);
  }

  // Pre-allocate output containers
  std::vector<TensorData> outputs(pipeline_depth);

  for (auto _ : state) {
    std::vector<std::future<InferErrorCode>> futures;
    futures.reserve(pipeline_depth);

    // Submit all tasks (non-blocking)
    for (int i = 0; i < pipeline_depth; ++i) {
      futures.push_back(stream_pool[i]->inferAsync(input, outputs[i]));
    }

    // Wait for all completions
    for (auto &f : futures) {
      if (f.get() != InferErrorCode::SUCCESS) {
        state.SkipWithError("Pipeline inference failed");
        return;
      }
    }
  }

  state.SetLabel("Depth=" + std::to_string(pipeline_depth));
  state.SetItemsProcessed(state.iterations() * pipeline_depth);
  state.counters["Throughput"] = benchmark::Counter(
      state.iterations() * pipeline_depth, benchmark::Counter::kIsRate,
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
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                 DataType::FLOAT32);

  for (auto _ : state) {
    // Create fresh stream for each iteration
    auto stream = async_engine->createExecutionContext();
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
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                 DataType::FLOAT32);

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
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input640 = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                    DataType::FLOAT32);
  auto input320 = createPinnedInput(async_engine.get(), {1, 3, 320, 320},
                                    DataType::FLOAT32);

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
  auto async_engine = EngineManager::instance().getAsyncEngine();

  for (auto _ : state) {
    auto stream = async_engine->createExecutionContext();
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
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                 DataType::FLOAT32);

  for (auto _ : state) {
    auto stream = async_engine->createExecutionContext();
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
  const bool graph_enabled = state.range(0) == 1;
  const bool pinned_memory = state.range(1) == 1;

  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  stream->setGraphEnabled(graph_enabled);

  TensorData input;
  if (pinned_memory) {
    input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                              DataType::FLOAT32);
  } else {
    input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);
  }

  warmupStream(stream.get(), input);

  for (auto _ : state) {
    TensorData output;
    stream->inferAsync(input, output).get();
  }

  std::string label = std::string(graph_enabled ? "Graph" : "NoGraph") + "_" +
                      (pinned_memory ? "Pinned" : "Pageable");
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
