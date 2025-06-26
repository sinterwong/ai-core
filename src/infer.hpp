/**
 * @file infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_HPP_
#define __INFERENCE_HPP_

#include "ai_core/types/algo_data_types.hpp"
#include "ai_core/types/infer_common_types.hpp"
#include "ai_core/types/infer_error_code.hpp"
#include "ai_core/types/model_output.hpp"

namespace ai_core::dnn {
class Inference {
public:
  Inference() = default;

  virtual ~Inference() {}

  virtual InferErrorCode initialize() = 0;

  virtual InferErrorCode infer(AlgoInput &input, ModelOutput &modelOutput) = 0;

  virtual InferErrorCode terminate() = 0;

  virtual const ModelInfo &getModelInfo() = 0;

  virtual void prettyPrintModelInfos();

protected:
  std::shared_ptr<ModelInfo> modelInfo;
};
} // namespace ai_core::dnn
#endif
