/**
 * @file algo_postproc_impl.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __ALGO_POSTPROC_IMPL_HPP__
#define __ALGO_POSTPROC_IMPL_HPP__

#include "ai_core/algo_postproc.hpp"
#include "postproc_base.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPostproc::Impl {
public:
  Impl(const std::string &moduleName, const AlgoPostprocParams &postprocParams);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode process(const TensorData &modelOutput,
                         AlgoPreprocParams &preprocParams, AlgoOutput &output,
                         AlgoPostprocParams &postprocParams);

  InferErrorCode terminate();

private:
  std::string moduleName_;
  AlgoPostprocParams postprocParams_;
  std::shared_ptr<PostprocssBase> postprocessor_;
};
} // namespace ai_core::dnn
#endif
