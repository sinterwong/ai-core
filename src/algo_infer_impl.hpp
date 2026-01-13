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
#ifndef AI_CORE_INFERENCE_VISION_INFER_HPP
#define AI_CORE_INFERENCE_VISION_INFER_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/algo_inference.hpp"
#include "ai_core/infer_engine_wrapper.hpp"
#include "ai_core/algo_postprocessor.hpp"
#include "ai_core/algo_preprocessor.hpp"
#include "ai_core/common_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/infer_config.hpp"
#include <memory>

namespace ai_core::dnn {
class AlgoInference::Impl {
public:
  Impl(const AlgoModuleTypes &algo_module_types,
       const AlgoInferParams &infer_params);

  ~Impl() = default;

  InferErrorCode initialize();

  InferErrorCode infer(const AlgoInput &input,
                       const AlgoPreprocParams &preproc_params,
                       const AlgoPostprocParams &postproc_params,
                       AlgoOutput &output);

  InferErrorCode batchInfer(const std::vector<AlgoInput> &inputs,
                            const AlgoPreprocParams &preproc_params,
                            const AlgoPostprocParams &postproc_params,
                            std::vector<AlgoOutput> &outputs);

  InferErrorCode terminate();

  const ModelInfo &getModelInfo() const noexcept;

  const AlgoModuleTypes &getModuleTypes() const noexcept;

private:
  AlgoModuleTypes m_algoModuleTypes;

  AlgoInferParams m_inferParams;
  std::shared_ptr<AlgoPreproc> m_preprocessor;
  std::shared_ptr<AlgoInferEngine> m_engine;
  std::shared_ptr<AlgoPostproc> m_postprocessor;
};
} // namespace ai_core::dnn
#endif
