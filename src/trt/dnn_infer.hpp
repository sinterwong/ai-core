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
#ifndef M_TENSORRT_INFERENCE_H_
#define M_TENSORRT_INFERENCE_H_

#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_base.hpp"
#include "ai_core/infer_params_types.hpp"
#include "trt_device_buffer.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <logger.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_core::dnn {

// logger for trt that forwards messages to our framework's logger
class TrtFrameworkLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      LOG_FATALS << "[TRT] " << msg;
      break;
    case Severity::kERROR:
      LOG_ERRORS << "[TRT] " << msg;
      break;
    case Severity::kWARNING:
      LOG_WARNINGS << "[TRT] " << msg;
      break;
    case Severity::kINFO:
      LOG_INFOS << "[TRT] " << msg;
      break;
    case Severity::kVERBOSE:
      LOG_INFOS << "[TRT] " << msg;
      break;
    default:
      LOG_INFOS << "[TRT] " << msg;
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
  // helper to calculate volume of dimensions
  static int64_t calculateVolume(const nvinfer1::Dims &dims);

  // initialization helpers
  InferErrorCode loadEngineFromPath(const std::string &path,
                                    bool needs_decrypt);
  InferErrorCode setupBindings();

  void releaseResources();

  AlgoInferParams mParams;
  TrtFrameworkLogger mLogger;

  // trt core components
  std::unique_ptr<nvinfer1::IRuntime> mRuntime;
  std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
  std::unique_ptr<nvinfer1::IExecutionContext> mContext;

  cudaStream_t mStream{nullptr};

  // Owns the actual device memory for all I/O tensors.
  std::vector<trt_utils::TrtDeviceBuffer> mManagedBuffers;

  // Maps tensor names to their corresponding device pointers.
  std::unordered_map<std::string, void *> mTensorAddressMap;

  // Maps tensor names to their size in bytes.
  std::unordered_map<std::string, size_t> mTensorSizeMap;

  std::unordered_set<std::string> mDynamicInputTensorNames;

  bool mIsInitialized{false};
  mutable std::mutex mMutex;
};

} // namespace ai_core::dnn
#endif
