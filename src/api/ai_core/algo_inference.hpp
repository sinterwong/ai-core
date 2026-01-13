/**
 * @file algo_inference.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_ALGO_INFER_BASE_HPP
#define AI_CORE_ALGO_INFER_BASE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/common_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/infer_config.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoInference {
public:
  AlgoInference(const AlgoModuleTypes &module_types,
                const AlgoInferParams &infer_params);

  ~AlgoInference();

  InferErrorCode initialize();

  InferErrorCode infer(const AlgoInput &input,
                       const AlgoPreprocParams &preproc_params,
                       const AlgoPostprocParams &postproc_params,
                       AlgoOutput &output);

  InferErrorCode batchInfer(const std::vector<AlgoInput> &inputs,
                            const AlgoPreprocParams &preproc_params,
                            const AlgoPostprocParams &postproc_params,
                            std::vector<AlgoOutput> &outputs);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

  const AlgoModuleTypes &getModuleTypes() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
