/**
 * @file algo_preproc.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __AI_CORE_ALGO_PREPROC_HPP__
#define __AI_CORE_ALGO_PREPROC_HPP__

#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPreproc {
public:
  AlgoPreproc(const std::string &moduleName);

  ~AlgoPreproc();

  InferErrorCode initialize();

  InferErrorCode process(AlgoInput &input, AlgoPreprocParams &preprocParams,
                         TensorData &modelInput);

  InferErrorCode terminate();

private:
  class Impl;
  std::unique_ptr<Impl> pImpl;
};
} // namespace ai_core::dnn
#endif // __AI_CORE_ALGO_PREPROC_HPP__
