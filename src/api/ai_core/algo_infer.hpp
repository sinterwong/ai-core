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

#include "types/algo_data_types.hpp"
#include "types/infer_common_types.hpp"
#include "types/infer_error_code.hpp"
#include "types/infer_params_types.hpp"

namespace ai_core::dnn {

class AlgoInference {
public:
  AlgoInference(const AlgoModuleTypes &moduleTypes,
                const AlgoInferParams &inferParams);

  ~AlgoInference();

  InferErrorCode initialize();

  InferErrorCode infer(AlgoInput &input, AlgoPreprocParams &preprocParams,
                       AlgoOutput &output, AlgoPostprocParams &postprocParams);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

  const AlgoModuleTypes &getModuleTypes() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};
} // namespace ai_core::dnn
#endif // __AI_CORE_ALGO_INFER_BASE_HPP__
