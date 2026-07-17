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
#ifndef AI_CORE_ONNXRUNTIME_INFERENCE_HPP
#define AI_CORE_ONNXRUNTIME_INFERENCE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/i_infer_engine.hpp"
#include "ai_core/infer_config.hpp"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <shared_mutex>
#include <unordered_set>

namespace ai_core::dnn {
class OrtAlgoInference : public IInferEnginePlugin {
public:
  explicit OrtAlgoInference(const AlgoConstructParams &params)
      : m_params(std::move(params.getParam<AlgoInferParams>("params"))) {}

  virtual ~OrtAlgoInference() override {}

  InferErrorCode initialize() override;
  InferErrorCode infer(const TensorData &inputs, TensorData &outputs) override;
  const ModelInfo &getModelInfo() override;
  InferErrorCode terminate() override;

private:
  static ONNXTensorElementDataType aiCoreDataTypeToOrt(DataType type);
  static DataType ortDataTypeToAiCore(ONNXTensorElementDataType type);

private:
  AlgoInferParams m_params;
  std::vector<std::string> m_inputNames;
  std::vector<std::string> m_outputNames;

  std::unique_ptr<Ort::Env> m_env;
  std::unique_ptr<Ort::Session> m_session;
  std::unique_ptr<Ort::MemoryInfo> m_memoryInfo;

  std::shared_ptr<ModelInfo> m_modelInfo;

  std::unordered_set<std::string> m_dynamicInputTensorNames;

  // Guards session lifetime: infer/getModelInfo take shared ownership
  // (Ort::Session::Run is thread-safe), initialize/terminate take exclusive.
  mutable std::shared_mutex m_mutex;
};
} // namespace ai_core::dnn
#endif
