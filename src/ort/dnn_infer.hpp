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
#include "ai_core/infer_params_types.hpp"
#include "infer_base.hpp"

#include <onnxruntime_cxx_api.h>

namespace ai_core::dnn {
class OrtAlgoInference : public InferBase {
public:
  explicit OrtAlgoInference(const AlgoConstructParams &params)
      : params_(std::move(params.getParam<AlgoInferParams>("params"))) {}

  virtual ~OrtAlgoInference() override {}

  virtual InferErrorCode initialize() override;

  virtual InferErrorCode infer(const TensorData &inputs,
                               TensorData &outputs) override;

  virtual const ModelInfo &getModelInfo() override;

  virtual InferErrorCode terminate() override;

protected:
  AlgoInferParams params_;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;

  std::vector<std::vector<int64_t>> inputShapes;
  std::vector<std::vector<int64_t>> outputShapes;

  // infer engine
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
  std::unique_ptr<Ort::MemoryInfo> memoryInfo;

  mutable std::mutex mtx_;
};
} // namespace ai_core::dnn
#endif
