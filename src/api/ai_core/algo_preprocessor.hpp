/**
 * @file algo_preprocessor.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_ALGO_PREPROC_HPP
#define AI_CORE_ALGO_PREPROC_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPreproc {
public:
  AlgoPreproc(const std::string &module_name);

  ~AlgoPreproc();

  InferErrorCode initialize();

  InferErrorCode process(const AlgoInput &input,
                         const AlgoPreprocParams &preproc_params,
                         TensorData &model_input,
                         std::shared_ptr<RuntimeContext> &runtime_context);

  InferErrorCode batchProcess(const std::vector<AlgoInput> &input,
                              const AlgoPreprocParams &preproc_params,
                              TensorData &model_input,
                              std::shared_ptr<RuntimeContext> &runtime_context);

  InferErrorCode terminate();

private:
  class Impl;
  std::unique_ptr<Impl> m_pImpl;
};
} // namespace ai_core::dnn
#endif
