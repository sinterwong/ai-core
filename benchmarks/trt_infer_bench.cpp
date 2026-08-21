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

namespace config {
constexpr int k_warmup_iterations = 10;
constexpr const char *k_model_path = "assets/models/yolov11n_trt_fp16.engine";
constexpr const char *k_model_name = "yolov11n";

// Shapes use NCHW order.
const std::vector<std::vector<int64_t>> k_input_shapes = {
    {1, 3, 640, 640},
    {1, 3, 320, 320},
    {1, 3, 1280, 1280},
};

const std::vector<int> k_pipeline_depths = {2, 3, 4, 6};

const std::vector<int> k_thread_counts = {1, 2, 4, 8};
} // namespace config

static size_t calculateSizeBytes(const std::vector<int64_t> &shape,
                                 DataType dtype) {
  size_t elements = 1;
  for (auto dim : shape)
    elements *= dim;

  size_t element_size = 4;
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

class EngineManager {
public:
  static EngineManager &instance() {
    // Intentionally leaked: CUDA objects may otherwise outlive the runtime
    // during process shutdown and make benchmark teardown nondeterministic.
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

static void setCommonCounters(benchmark::State &state,
                              const std::vector<int64_t> &shape,
                              int items_per_iteration = 1) {
  size_t input_bytes = calculateSizeBytes(shape, DataType::FLOAT32);
  // YOLO 640x640 emits one 1x84x8400 tensor.
  size_t output_bytes = 84 * 8400 * sizeof(float);
  size_t total_bytes = input_bytes + output_bytes;

  state.SetItemsProcessed(state.iterations() * items_per_iteration);
  state.SetBytesProcessed(state.iterations() * total_bytes *
                          items_per_iteration);

  state.counters["InputMB"] = input_bytes / (1024.0 * 1024.0);
  state.counters["Latency_us"] = benchmark::Counter(
      state.iterations(),
      benchmark::Counter::kIsRate | benchmark::Counter::kInvert,
      benchmark::Counter::OneK::kIs1000);
}

static void BM_TRT_Baseline_Sync(benchmark::State &state) {
  auto engine = EngineManager::instance().getEngine();
  auto input = createPageableInput({1, 3, 640, 640}, DataType::FLOAT32);

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

// Google Benchmark runs the body concurrently against the shared engine. Each
// call borrows an execution context, so throughput should scale with threads.
static void BM_TRT_Sync_Concurrent(benchmark::State &state) {
  auto engine = EngineManager::instance().getEngine();
  // Each worker owns its tensors; only the engine is shared.
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

// Report an explicit aggregate rate because the standard multi-threaded
// benchmark aggregation can obscure context-pool scaling.
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

static void BM_TRT_Async_WithGraph_Pageable(benchmark::State &state) {
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  stream->setGraphEnabled(true);

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
BENCHMARK(BM_TRT_Async_WithGraph_Pageable)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

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

static void BM_TRT_MemoryType_Comparison(benchmark::State &state) {
  const bool use_pinned = state.range(0) == 1;

  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  // Disable graphs so the benchmark isolates host-memory transfer behavior.
  stream->setGraphEnabled(false);

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

static void BM_TRT_Pipeline_Throughput(benchmark::State &state) {
  const int pipeline_depth = state.range(0);

  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream_pool = async_engine->createContextPool(pipeline_depth);

  for (auto &s : stream_pool) {
    s->setGraphEnabled(true);
  }

  auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                 DataType::FLOAT32);

  for (auto &s : stream_pool) {
    warmupStream(s.get(), input, 5);
  }

  std::vector<TensorData> outputs(pipeline_depth);

  for (auto _ : state) {
    std::vector<std::future<InferErrorCode>> futures;
    futures.reserve(pipeline_depth);

    for (int i = 0; i < pipeline_depth; ++i) {
      futures.push_back(stream_pool[i]->inferAsync(input, outputs[i]));
    }

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

static void BM_TRT_Graph_Capture_Overhead(benchmark::State &state) {
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                 DataType::FLOAT32);

  for (auto _ : state) {
    // A fresh context forces every iteration to include graph capture.
    auto stream = async_engine->createExecutionContext();
    stream->setGraphEnabled(true);

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

static void BM_TRT_Graph_Replay_Latency(benchmark::State &state) {
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                 DataType::FLOAT32);

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

static void BM_TRT_Graph_Recapture_Overhead(benchmark::State &state) {
  auto async_engine = EngineManager::instance().getAsyncEngine();
  auto stream = async_engine->createExecutionContext();
  stream->setGraphEnabled(true);

  auto input640 = createPinnedInput(async_engine.get(), {1, 3, 640, 640},
                                    DataType::FLOAT32);
  auto input320 = createPinnedInput(async_engine.get(), {1, 3, 320, 320},
                                    DataType::FLOAT32);

  warmupStream(stream.get(), input640);

  bool toggle = false;
  for (auto _ : state) {
    // Alternating shapes invalidates and recaptures the graph each time.
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
