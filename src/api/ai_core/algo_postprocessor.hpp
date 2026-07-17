/**
 * @file algo_postprocessor.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_ALGO_POSTPROC_HPP
#define AI_CORE_ALGO_POSTPROC_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPostproc {
public:
  AlgoPostproc(const std::string &module_name);

  ~AlgoPostproc();

  /**
   * @brief Create the plugin and bind + validate the postprocess parameters
   * once. process() calls carry data only.
   */
  InferErrorCode initialize(const AlgoPostprocParams &postproc_params);

  /**
   * @param postproc_override optional per-call parameter override; pass
   * nullptr to use the parameters bound at initialize().
   */
  InferErrorCode process(const TensorData &model_output, AlgoOutput &output,
                         std::shared_ptr<RuntimeContext> &runtime_context,
                         const AlgoPostprocParams *postproc_override = nullptr);

  InferErrorCode
  batchProcess(const TensorData &model_output, std::vector<AlgoOutput> &output,
               std::shared_ptr<RuntimeContext> &runtime_context,
               const AlgoPostprocParams *postproc_override = nullptr);

  InferErrorCode terminate();

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
