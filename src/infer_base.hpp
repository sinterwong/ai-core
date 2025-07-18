/**
 * @file infer_base.hpp
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

#include <memory>

#include "ai_core/infer_common_types.hpp"
#include "ai_core/infer_error_code.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {
class InferBase {
public:
  InferBase() = default;

  virtual ~InferBase() {}

  virtual InferErrorCode initialize() = 0;

  virtual InferErrorCode infer(const TensorData &inputs,
                               TensorData &outputs) = 0;

  virtual InferErrorCode terminate() = 0;

  virtual const ModelInfo &getModelInfo() = 0;

  virtual void prettyPrintModelInfos();

protected:
  std::shared_ptr<ModelInfo> modelInfo;
};
} // namespace ai_core::dnn
#endif
