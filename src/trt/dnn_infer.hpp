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
#include "ai_core/infer_params_types.hpp"
#include "infer_base.hpp"

namespace ai_core::dnn {
class TrtAlgoInference : public InferBase {
public:
  explicit TrtAlgoInference(const AlgoConstructParams &params);

  virtual ~TrtAlgoInference() override {}

  virtual InferErrorCode initialize() override;

  virtual InferErrorCode infer(TensorData &inputs,
                               TensorData &outputs) override;

  virtual const ModelInfo &getModelInfo() override;

  virtual InferErrorCode terminate() override;

protected:
  AlgoInferParams params_;
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;

  std::vector<std::vector<int64_t>> inputShapes_;
  std::vector<std::vector<int64_t>> outputShapes_;

  mutable std::mutex mtx_;
};
} // namespace ai_core::dnn
#endif
