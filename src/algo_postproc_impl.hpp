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
#ifndef ALGO_POSTPROC_IMPL_HPP
#define ALGO_POSTPROC_IMPL_HPP

#include "ai_core/algo_postprocessor.hpp"
#include "ai_core/i_postprocess.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPostproc::Impl {
public:
  Impl(const std::string &module_name);

  ~Impl() = default;

  InferErrorCode initialize(const AlgoPostprocParams &postproc_params);

  InferErrorCode process(const TensorData &model_output, AlgoOutput &output,
                         std::shared_ptr<RuntimeContext> &runtime_context,
                         const AlgoPostprocParams *postproc_override);

  InferErrorCode batchProcess(const TensorData &model_output,
                              std::vector<AlgoOutput> &output,
                              std::shared_ptr<RuntimeContext> &runtime_context,
                              const AlgoPostprocParams *postproc_override);

  InferErrorCode terminate();

private:
  std::string m_moduleName;
  AlgoPostprocParams m_boundParams;
  std::shared_ptr<IPostprocessPlugin> m_postprocessor;
};
} // namespace ai_core::dnn
#endif
