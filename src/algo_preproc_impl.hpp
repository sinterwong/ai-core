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
  Impl(const std::string &moduleName, const AlgoPreprocParams &preprocParams);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode process(AlgoInput &input, AlgoPreprocParams &preprocParams,
                         TensorData &modelInput);

  InferErrorCode terminate();

private:
  std::string moduleName_;
  AlgoPreprocParams preprocParams_;
  std::shared_ptr<PreprocssBase> preprocessor_;
};
} // namespace ai_core::dnn
#endif
