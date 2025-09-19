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
#ifndef __ALGO_PREPROC_IMPL_HPP__
#define __ALGO_PREPROC_IMPL_HPP__

#include "ai_core/algo_preproc.hpp"
#include "preproc_base.hpp"
#include <memory>

namespace ai_core::dnn {

class AlgoPreproc::Impl {
public:
  Impl(const std::string &moduleName);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode process(const AlgoInput &input,
                         const AlgoPreprocParams &preprocParams,
                         TensorData &modelInput,
                         std::shared_ptr<RuntimeContext> &runtimeContext);

  InferErrorCode batchProcess(const std::vector<AlgoInput> &input,
                              const AlgoPreprocParams &preprocParams,
                              TensorData &modelInput,
                              std::shared_ptr<RuntimeContext> &runtimeContext);

  InferErrorCode terminate();

private:
  std::string moduleName_;
  std::shared_ptr<PreprocssBase> preprocessor_;
};
} // namespace ai_core::dnn
#endif
