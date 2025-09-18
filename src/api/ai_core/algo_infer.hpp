/**
 * @file algo_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __AI_CORE_ALGO_INFER_BASE_HPP__
#define __AI_CORE_ALGO_INFER_BASE_HPP__

#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_common_types.hpp"
#include "ai_core/infer_error_code.hpp"
#include "ai_core/infer_params_types.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoInference {
public:
  AlgoInference(const AlgoModuleTypes &moduleTypes,
                const AlgoInferParams &inferParams);

  ~AlgoInference();

  InferErrorCode initialize();

  InferErrorCode infer(const AlgoInput &input,
                       const AlgoPreprocParams &preprocParams,
                       const AlgoPostprocParams &postprocParams,
                       AlgoOutput &output);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

  const AlgoModuleTypes &getModuleTypes() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};
} // namespace ai_core::dnn
#endif // __AI_CORE_ALGO_INFER_BASE_HPP__
