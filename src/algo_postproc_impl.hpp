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
#include "ai_core/postproc_base.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPostproc::Impl {
public:
  Impl(const std::string &moduleName);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode process(const TensorData &modelOutput, AlgoOutput &output,
                         const AlgoPostprocParams &postprocParams,
                         std::shared_ptr<RuntimeContext> &runtimeContext);

  InferErrorCode batchProcess(const TensorData &modelOutput,
                              std::vector<AlgoOutput> &output,
                              const AlgoPostprocParams &postprocParams,
                              std::shared_ptr<RuntimeContext> &runtimeContext);

  InferErrorCode terminate();

private:
  std::string moduleName_;
  std::shared_ptr<IPostprocssPlugin> postprocessor_;
};
} // namespace ai_core::dnn
#endif
