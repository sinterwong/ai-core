#ifndef AI_CORE_TRT_INFER_STREAM_HPP
#define AI_CORE_TRT_INFER_STREAM_HPP

#include "ai_core/i_execution_context.hpp"
#include "cuda_device_buffer.cuh"
#include "cuda_host_buffer.cuh"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <future>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ai_core::dnn {

class TrtAlgoInference;

/**
 * @brief Single-owner TensorRT execution context with isolated CUDA resources.
 *
 * Device buffers, pinned output buffers, the CUDA stream, and any captured
 * graph belong to this context. Enabling graph mode captures on the next
 * inference and replays thereafter. Graph mode requires static input shapes
 * and stable buffer addresses.
 *
 * @par Thread safety
 * A stream is not thread-safe and must be owned by one worker. Distinct streams
 * from the same engine may execute concurrently.
 */
class TrtInferStream : public IExecutionContext {
public:
  /**
   * @brief Construct a context sharing immutable engine resources.
   */
  explicit TrtInferStream(TrtAlgoInference &engine);

  ~TrtInferStream() override;

  TrtInferStream(const TrtInferStream &) = delete;
  TrtInferStream &operator=(const TrtInferStream &) = delete;
  TrtInferStream(TrtInferStream &&) = delete;
  TrtInferStream &operator=(TrtInferStream &&) = delete;

  /**
   * @brief Enqueue host-to-device copy, inference, and device-to-host copy.
   *
   * The returned deferred future synchronizes this CUDA stream and finalizes
   * `outputs` when `get()` is called. Inputs and outputs must satisfy the base
   * interface lifetime contract until then.
   */
  std::future<InferErrorCode> inferAsync(const TensorData &inputs,
                                         TensorData &outputs) override;

  InferErrorCode synchronize() override;

  bool isComplete() const override;

  BackendHandle getHandle() const override;

  /**
   * @brief Enable or disable CUDA graph capture and replay.
   *
   * Enabling fails for dynamic input shapes. Disabling destroys any captured
   * graph and resumes normal `enqueueV3` execution.
   */
  InferErrorCode setGraphEnabled(bool enable) override;

  bool isGraphEnabled() const override;

  /** Allocate the CUDA stream, TensorRT context, and reusable I/O buffers. */
  InferErrorCode initialize();

private:
  InferErrorCode allocateBuffers();
  InferErrorCode allocatePinnedOutputBuffers();

  bool updateInputShapesIfNeeded(const TensorData &inputs);
  InferErrorCode copyInputsToDevice(const TensorData &inputs);

  /**
   * @brief Enqueue device-to-host copies without synchronizing the stream.
   */
  InferErrorCode submitAsyncD2H(TensorData &outputs);

  /**
   * @brief Copy pinned staging data after the CUDA stream has synchronized.
   */
  InferErrorCode finalizeOutputs(TensorData &outputs);

  /** Copy device outputs synchronously on the legacy path. */
  InferErrorCode copyOutputsToHost(TensorData &outputs);

  InferErrorCode executeInference();
  InferErrorCode captureGraph();
  InferErrorCode launchGraph();
  void destroyGraph();

  TrtAlgoInference &m_engine;

  // Declaration order makes `m_context` die before the shared engine handle.
  std::shared_ptr<nvinfer1::ICudaEngine> m_sharedEngine;

  cudaStream_t m_cudaStream{nullptr};

  std::unique_ptr<nvinfer1::IExecutionContext> m_context;

  std::vector<cuda_utils::DeviceByteBuffer> m_deviceBuffers;
  std::unordered_map<std::string, void *> m_tensorAddressMap;

  std::unordered_map<std::string, cuda_utils::CudaHostBuffer<uint8_t>>
      m_pinnedOutputBuffers;

  // Avoid redundant TensorRT shape updates on this context.
  std::unordered_map<std::string, std::vector<int64_t>> m_cachedInputShapes;

  bool m_graphEnabled{false};
  bool m_graphCaptured{false};
  cudaGraph_t m_cudaGraph{nullptr};
  cudaGraphExec_t m_cudaGraphExec{nullptr};

  bool m_initialized{false};
};

} // namespace ai_core::dnn

#endif // AI_CORE_TRT_INFER_STREAM_HPP
