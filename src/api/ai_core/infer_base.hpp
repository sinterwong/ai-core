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
#ifndef AI_CORE_INFER_BASE_HPP
#define AI_CORE_INFER_BASE_HPP

#include "ai_core/infer_common_types.hpp"
#include "ai_core/infer_error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

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
