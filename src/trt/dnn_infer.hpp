/**
 * @file dnn_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_TENSORRT_INFERENCE_HPP
#define AI_CORE_TENSORRT_INFERENCE_HPP

#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_base.hpp"
#include "ai_core/infer_params_types.hpp"
#include "ai_core/logger.hpp"
#include "cuda_device_buffer.cuh"
#include "cuda_host_buffer.cuh"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_core::dnn {

class TrtFrameworkLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      LOG_FATALS << "[TRT] " << msg;
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

class TrtAlgoInference : public IInferEnginePlugin {
public:
  explicit TrtAlgoInference(const AlgoConstructParams &params);
  ~TrtAlgoInference() override;

  InferErrorCode initialize() override;
  InferErrorCode infer(const TensorData &inputs, TensorData &outputs) override;
  const ModelInfo &getModelInfo() override;
  InferErrorCode terminate() override;

private:
  // Helper to calculate volume of dimensions
  static int64_t calculateVolume(const nvinfer1::Dims &dims);

  // Initialization helpers
  InferErrorCode loadEngineFromPath(const std::string &path,
                                    bool needs_decrypt);
  InferErrorCode setupBindings();
  InferErrorCode setupPinnedOutputBuffers();
  InferErrorCode setupCudaGraph();

  void releaseResources();

  // Optimized inference path
  InferErrorCode inferWithGraph(const TensorData &inputs, TensorData &outputs);
  InferErrorCode inferWithoutGraph(const TensorData &inputs,
                                   TensorData &outputs);

  // Helper methods
  bool updateInputShapesIfNeeded(const TensorData &inputs);
  void copyInputsToDevice(const TensorData &inputs);
  void copyOutputsToHost(TensorData &outputs);

  AlgoInferParams mParams;
  TrtFrameworkLogger mLogger;

  // TRT core components
  std::unique_ptr<nvinfer1::IRuntime> mRuntime;
  std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
  std::unique_ptr<nvinfer1::IExecutionContext> mContext;

  // High-priority non-blocking stream for inference
  cudaStream_t mStream{nullptr};

  // Owns the actual device memory for all I/O tensors
  std::vector<cuda_utils::DeviceByteBuffer> mManagedBuffers;

  // Maps tensor names to their corresponding device pointers
  std::unordered_map<std::string, void *> mTensorAddressMap;

  // Maps tensor names to their size in bytes
  std::unordered_map<std::string, size_t> mTensorSizeMap;

  std::unordered_set<std::string> mDynamicInputTensorNames;

  std::unordered_map<std::string, cuda_utils::CudaHostBuffer<uint8_t>>
      mPinnedOutputBuffers;

  cudaGraph_t mCudaGraph{nullptr};
  cudaGraphExec_t mCudaGraphExec{nullptr};
  bool mCudaGraphEnabled{false};
  bool mCudaGraphCaptured{false};

  // Track if graph capture is in progress (for multi-thread safety diagnostics)
  std::atomic<bool> mGraphCaptureInProgress{false};

  // Cache last used input shapes to avoid redundant setInputShape calls.
  // TensorRT's setInputShape has non-trivial overhead.
  std::unordered_map<std::string, std::vector<int64_t>> mCachedInputShapes;

  // If all inputs have static shapes, we can use CUDA Graph for acceleration
  bool mAllInputsStatic{true};

  bool mIsInitialized{false};
  mutable std::mutex mMutex;

  // Model metadata
  std::shared_ptr<ModelInfo> modelInfo;
};

} // namespace ai_core::dnn

#endif
