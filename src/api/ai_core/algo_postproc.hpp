/**
 * @file algo_postproc.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __AI_CORE_ALGO_POSTPROC_HPP__
#define __AI_CORE_ALGO_POSTPROC_HPP__

#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPostproc {
public:
  AlgoPostproc(const std::string &moduleName);

  ~AlgoPostproc();

  InferErrorCode initialize();

  InferErrorCode process(const TensorData &modelOutput,
                         AlgoPreprocParams &preprocParams, AlgoOutput &output,
                         AlgoPostprocParams &postprocParams);

  InferErrorCode terminate();

private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};
} // namespace ai_core::dnn
#endif // __AI_CORE_ALGO_POSTPROC_HPP__
