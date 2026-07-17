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

  /**
   * @brief Initialize the pipeline, binding + validating the pre/postprocess
   * parameters once. infer() calls carry data only.
   */
  InferErrorCode initialize(const AlgoPreprocParams &preproc_params,
                            const AlgoPostprocParams &postproc_params);

  /**
   * @brief Run one inference. The overrides are optional per-call parameter
   * replacements; pass nullptr to use the parameters bound at initialize().
   */
  InferErrorCode infer(const AlgoInput &input, AlgoOutput &output,
                       const AlgoPreprocParams *preproc_override = nullptr,
                       const AlgoPostprocParams *postproc_override = nullptr);

  InferErrorCode
  batchInfer(const std::vector<AlgoInput> &inputs,
             std::vector<AlgoOutput> &outputs,
             const AlgoPreprocParams *preproc_override = nullptr,
             const AlgoPostprocParams *postproc_override = nullptr);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

  const AlgoModuleTypes &getModuleTypes() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
