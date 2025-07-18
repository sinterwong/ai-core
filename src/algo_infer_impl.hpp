/**
 * @file algo_infer_impl.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-04-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_INFER_HPP__
#define __INFERENCE_VISION_INFER_HPP__

#include "ai_core/algo_data_types.hpp"
#include "ai_core/algo_infer.hpp"
#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/algo_postproc.hpp"
#include "ai_core/algo_preproc.hpp"
#include "ai_core/infer_common_types.hpp"
#include "ai_core/infer_error_code.hpp"
#include "ai_core/infer_params_types.hpp"
#include <memory>

namespace ai_core::dnn {
class AlgoInference::Impl {
public:
  Impl(const AlgoModuleTypes &algoModuleTypes,
       const AlgoInferParams &inferParams);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode infer(AlgoInput &input, AlgoPreprocParams &preprocParams,
                       AlgoOutput &output, AlgoPostprocParams &postprocParams);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

  const AlgoModuleTypes &getModuleTypes() const noexcept;

private:
  AlgoModuleTypes algoModuleTypes_;

  AlgoInferParams inferParams_;
  std::shared_ptr<AlgoPreproc> preprocessor_;
  std::shared_ptr<AlgoInferEngine> engine_;
  std::shared_ptr<AlgoPostproc> postprocessor_;
};
} // namespace ai_core::dnn
#endif
