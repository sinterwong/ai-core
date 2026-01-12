/**
 * @file algo_preproc_impl.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef ALGO_PREPROC_IMPL_HPP
#define ALGO_PREPROC_IMPL_HPP

#include "ai_core/algo_preproc.hpp"
#include "ai_core/preproc_base.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPreproc::Impl {
public:
  Impl(const std::string &module_name);

  ~Impl() = default;

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
  std::string m_moduleName;
  std::shared_ptr<IPreprocssPlugin> m_preprocessor;
};
} // namespace ai_core::dnn
#endif
