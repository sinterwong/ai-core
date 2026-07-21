/**
 * @file async_pipeline_main.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief End-to-end async inference pipeline: the framework's performance
 * showcase. Demonstrates the supported path to concurrency —
 *
 *   AlgoInferEngine::getAsyncEngine()   (front door, no dynamic_cast)
 *     -> a pool of IExecutionContext    (one CUDA stream each)
 *     -> pinned host buffers            (async, overlappable H2D/D2H)
 *     -> CUDA Graph                     (amortized launch overhead)
 *     -> N worker threads pulling from a shared frame queue
 *
 * Each worker owns one execution context (contexts are NOT thread-safe) and
 * runs inferAsync + synchronize, so compute on different streams overlaps.
 *
 * @version 0.1
 * @date 2026-07-18
 *
 * @copyright Copyright (c) 2026
 */
#include "ai_core/algo_types.hpp"
#include "ai_core/infer_async.hpp"
#include "ai_core/infer_config.hpp"
#include "ai_core/infer_engine_wrapper.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/tensor_data.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace ai_core;
using namespace ai_core::dnn;

namespace {

constexpr int kModelWidth = 640;
constexpr int kModelHeight = 640;
constexpr int kModelChannels = 3;

size_t inputElementCount() {
  return static_cast<size_t>(kModelChannels) * kModelHeight * kModelWidth;
}

// Fill a pinned input buffer with a deterministic synthetic frame. In a real
// pipeline this is where preprocessing would write (e.g. the CPU/CUDA frame
// preprocessor targeting an accelerator buffer).
TensorData makePinnedInput(IAsyncInferEngine &engine, int seed) {
  const size_t elems = inputElementCount();
  TypedBuffer buffer = engine.allocateAcceleratorBuffer(DataType::FLOAT32,
                                                        elems * sizeof(float));
  float *ptr = buffer.getHostPtr<float>();
  for (size_t i = 0; i < elems; ++i) {
    ptr[i] = static_cast<float>((i + seed) % 255) / 255.0f;
  }

  TensorData data;
  data.set("images", std::move(buffer),
           {1, kModelChannels, kModelHeight, kModelWidth});
  return data;
}

} // namespace

int main(int argc, char **argv) {
  logging::Logger::instance().setLevel(logging::LogLevel::Info);

  std::string model_path = "assets/models/yolov11n_trt_fp16.engine";
  int num_workers = 4;
  int total_frames = 400;
  if (argc > 1)
    model_path = argv[1];
  if (argc > 2)
    num_workers = std::stoi(argv[2]);
  if (argc > 3)
    total_frames = std::stoi(argv[3]);

  // --- Build the engine and reach the async front door -----------------------
  AlgoInferParams infer_params;
  infer_params.name = "yolov11n_async";
  infer_params.model_path = model_path;
  infer_params.device_type = DeviceType::GPU;
  infer_params.data_type = DataType::FLOAT32;

  AlgoInferEngine engine("TrtAlgoInference", infer_params);
  if (engine.initialize() != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to initialize engine (need a TensorRT engine at "
                << model_path
                << "; build one with scripts/fetch_models.sh --trt-only).";
    return 1;
  }

  std::shared_ptr<IAsyncInferEngine> async_engine = engine.getAsyncEngine();
  if (!async_engine) {
    LOG_ERROR_S << "Backend has no async support (getAsyncEngine() == null). "
                   "This example needs the TensorRT backend.";
    return 1;
  }
  LOG_INFO_S << "Async engine acquired. Workers=" << num_workers
             << " frames=" << total_frames;

  // --- Build a pool of execution contexts (one CUDA stream each) -------------
  // Each context gets its own pinned input, CUDA graph, and is captured during
  // a warmup run so subsequent inferences replay the graph.
  std::vector<std::shared_ptr<IExecutionContext>> pool =
      async_engine->createContextPool(num_workers);
  std::vector<TensorData> pinned_inputs;
  pinned_inputs.reserve(num_workers);
  for (int i = 0; i < num_workers; ++i) {
    pool[i]->setGraphEnabled(true);
    pinned_inputs.push_back(makePinnedInput(*async_engine, i));
    // Warmup captures the CUDA graph for this context.
    TensorData warm_out;
    pool[i]->inferAsync(pinned_inputs[i], warm_out).get();
  }

  // --- Run the pipeline: N workers pull frame indices and infer --------------
  std::atomic<int> next_frame{0};
  std::atomic<int> failures{0};

  auto worker = [&](int id) {
    IExecutionContext &ctx = *pool[id];
    const TensorData &input = pinned_inputs[id];
    while (true) {
      int frame = next_frame.fetch_add(1, std::memory_order_relaxed);
      if (frame >= total_frames) {
        break;
      }
      TensorData output;
      if (ctx.inferAsync(input, output).get() != InferErrorCode::SUCCESS) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };

  auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  workers.reserve(num_workers);
  for (int i = 0; i < num_workers; ++i) {
    workers.emplace_back(worker, i);
  }
  for (auto &w : workers) {
    w.join();
  }
  auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();

  const int processed = total_frames - failures.load();
  LOG_INFO_S << "Processed " << processed << "/" << total_frames
             << " frames in " << elapsed << " s  (" << (processed / elapsed)
             << " img/s, " << num_workers << " workers)";
  if (failures.load() > 0) {
    LOG_ERROR_S << failures.load() << " inference(s) failed.";
    return 1;
  }
  return 0;
}
