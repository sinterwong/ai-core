#ifndef AI_CORE_TENSORRT_INFERENCE_HPP
#define AI_CORE_TENSORRT_INFERENCE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/infer_async.hpp"
#include "ai_core/infer_config.hpp"
#include "cuda_device_buffer.cuh"
#include "cuda_host_buffer.cuh"
#include "logger.hpp"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_core::dnn {

class TrtInferStream;

class TrtFrameworkLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      LOG_FATAL_S << "[TRT] " << msg;
      break;
    case Severity::kERROR:
      LOG_ERROR_S << "[TRT] " << msg;
      break;
    case Severity::kWARNING:
      LOG_WARNING_S << "[TRT] " << msg;
      break;
    case Severity::kINFO:
      LOG_INFO_S << "[TRT] " << msg;
      break;
    case Severity::kVERBOSE:
      LOG_DEBUG_S << "[TRT] " << msg;
      break;
    default:
      LOG_INFO_S << "[TRT] " << msg;
      break;
    }
  }
};

/**
 * @brief TensorRT backend with synchronous pooling and explicit async contexts.
 *
 * Runtime, engine, and model metadata are shared. Each `TrtInferStream` owns a
 * TensorRT execution context, CUDA stream, device buffers, pinned output
 * buffers, and optional CUDA graph.
 *
 * @par Thread safety
 * `infer()` borrows an execution context from a synchronized pool, so calls may
 * execute concurrently. `createExecutionContext()` is also thread-safe. Each
 * returned context remains single-owner and is not itself thread-safe.
 */
class TrtAlgoInference : public IAsyncInferEngine {
public:
  explicit TrtAlgoInference(const AlgoConstructParams &params);
  ~TrtAlgoInference() override;

  TrtAlgoInference(const TrtAlgoInference &) = delete;
  TrtAlgoInference &operator=(const TrtAlgoInference &) = delete;
  TrtAlgoInference(TrtAlgoInference &&) = delete;
  TrtAlgoInference &operator=(TrtAlgoInference &&) = delete;

  InferErrorCode initialize() override;

  /**
   * @brief Run synchronously on a context borrowed from the internal pool.
   */
  InferErrorCode infer(const TensorData &inputs, TensorData &outputs) override;

  const ModelInfo &getModelInfo() override;
  InferErrorCode terminate() override;

  /**
   * @brief Create an independently buffered TensorRT execution context.
   * @throws std::runtime_error If the engine is not initialized.
   */
  std::shared_ptr<IExecutionContext> createExecutionContext() override;

  /**
   * @brief Allocate owned CUDA-pinned host memory.
   */
  TypedBuffer allocateAcceleratorBuffer(DataType type,
                                        size_t size_bytes) override;

  /**
   * @brief Create a context with pinned buffers sized from model metadata.
   *
   * Allocates pinned buffers for all model inputs/outputs based on
   * max shapes from optimization profile.
   */
  ContextPackage createContextPackage() override;

private:
  friend class TrtInferStream;

  // Initialization helpers

  static int64_t calculateVolume(const nvinfer1::Dims &dims);
  InferErrorCode loadEngineFromPath(const std::string &path,
                                    bool needs_decrypt);
  InferErrorCode setupBindings();
  InferErrorCode setupPinnedOutputBuffers();
  void releaseResources();

  /** Execute on the legacy default context without CUDA graph capture. */
  InferErrorCode inferWithoutGraph(const TensorData &inputs,
                                   TensorData &outputs);

  bool updateInputShapesIfNeeded(const TensorData &inputs);
  void copyInputsToDevice(const TensorData &inputs);
  void copyOutputsToHost(TensorData &outputs);

  AlgoInferParams m_params;
  TrtFrameworkLogger m_logger;

  // Streams retain shared engine ownership so their contexts cannot outlive it.
  std::unique_ptr<nvinfer1::IRuntime> m_runtime;
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;

  // Shared metadata is immutable after initialization.
  std::shared_ptr<ModelInfo> m_modelInfo;

  std::unordered_set<std::string> m_dynamicInputTensorNames;
  std::unordered_map<std::string, size_t> m_tensorSizeMap;
  bool m_allInputsStatic{true};

  bool m_isInitialized{false};

  // Default-context resources used by `inferWithoutGraph()`.
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;
  cudaStream_t m_stream{nullptr};
  std::vector<cuda_utils::DeviceByteBuffer> m_managedBuffers;
  std::unordered_map<std::string, void *> m_tensorAddressMap;

  std::unordered_map<std::string, cuda_utils::CudaHostBuffer<uint8_t>>
      m_pinnedOutputBuffers;

  // Avoid redundant TensorRT shape updates on the default context.
  std::unordered_map<std::string, std::vector<int64_t>> m_cachedInputShapes;

  // Guards initialize()/terminate() lifecycle only.
  mutable std::mutex m_mutex;

  // The idle pool grows lazily to the peak number of concurrent `infer()`
  // callers and is guarded by `m_poolMutex`.
  std::mutex m_poolMutex;
  std::vector<std::shared_ptr<IExecutionContext>> m_idlePool;

  std::shared_ptr<IExecutionContext> acquireContext();
  void releaseContext(std::shared_ptr<IExecutionContext> ctx);
};

} // namespace ai_core::dnn

#endif // AI_CORE_TENSORRT_INFERENCE_HPP
