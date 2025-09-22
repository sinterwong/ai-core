/**
 * @file ort_dnn_infer_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __ONNXRUNTIME_INFERENCE_H_
#define __ONNXRUNTIME_INFERENCE_H_

#include <memory>

#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_base.hpp"
#include "ai_core/infer_params_types.hpp"

#include <onnxruntime_cxx_api.h>
#include <unordered_set>

namespace ai_core::dnn {
class OrtAlgoInference : public InferBase {
public:
  explicit OrtAlgoInference(const AlgoConstructParams &params)
      : mParams(std::move(params.getParam<AlgoInferParams>("params"))) {}

  virtual ~OrtAlgoInference() override {}

  InferErrorCode initialize() override;
  InferErrorCode infer(const TensorData &inputs, TensorData &outputs) override;
  const ModelInfo &getModelInfo() override;
  InferErrorCode terminate() override;

private:
  static ONNXTensorElementDataType aiCoreDataTypeToOrt(DataType type);
  static DataType ortDataTypeToAiCore(ONNXTensorElementDataType type);

private:
  AlgoInferParams mParams;
  std::vector<std::string> mInputNames;
  std::vector<std::string> mOutputNames;

  std::unique_ptr<Ort::Env> mEnv;
  std::unique_ptr<Ort::Session> mSession;
  std::unique_ptr<Ort::MemoryInfo> mMemoryInfo;

  std::shared_ptr<ModelInfo> modelInfo;

  std::unordered_set<std::string> mDynamicInputTensorNames;

  mutable std::mutex mMutex;
};
} // namespace ai_core::dnn
#endif
