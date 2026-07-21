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

#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/i_preprocess.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPreproc::Impl {
public:
  Impl(const std::string &module_name);

  ~Impl() = default;

  InferErrorCode initialize(const AlgoPreprocParams &preproc_params);

  InferErrorCode process(const AlgoInput &input, TensorData &model_input,
                         std::shared_ptr<RuntimeContext> &runtime_context,
                         const AlgoPreprocParams *preproc_override);

  InferErrorCode batchProcess(const std::vector<AlgoInput> &input,
                              TensorData &model_input,
                              std::shared_ptr<RuntimeContext> &runtime_context,
                              const AlgoPreprocParams *preproc_override);

  InferErrorCode terminate();

private:
  std::string m_moduleName;
  AlgoPreprocParams m_boundParams;
  std::shared_ptr<IPreprocessPlugin> m_preprocessor;
};
} // namespace ai_core::dnn
#endif
