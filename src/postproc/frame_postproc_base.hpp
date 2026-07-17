/**
 * @file frame_postproc_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Shared IPostprocessPlugin adapter for frame-based postprocessors
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_FRAME_POSTPROC_BASE_HPP
#define AI_CORE_FRAME_POSTPROC_BASE_HPP

#include "ai_core/algo_types.hpp"
#include "ai_core/i_postprocess.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {

/**
 * @brief Implements the IPostprocessPlugin boilerplate (param extraction,
 * output validation, preproc context retrieval) so concrete postprocessors
 * only implement the typed hooks.
 *
 * @tparam ParamsT the concrete params type held by AlgoPostprocParams
 * @tparam RequiresPrepContext whether the algorithm needs the
 * FrameTransformContext produced by the preprocessor (e.g. to map boxes back
 * to original image coordinates). When false, a default-constructed context
 * is passed to the hooks.
 */
template <typename ParamsT, bool RequiresPrepContext>
class FramePostprocBase : public IPostprocessPlugin {
public:
  InferErrorCode
  process(const TensorData &model_output, const AlgoPostprocParams &post_args,
          AlgoOutput &algo_output,
          std::shared_ptr<RuntimeContext> &runtime_context) const final {
    if (model_output.datas.empty()) {
      LOG_ERROR_S << "model_output.datas is empty";
      return InferErrorCode::InferOutputError;
    }

    auto params = post_args.getParams<ParamsT>();
    if (params == nullptr) {
      LOG_ERROR_S << "AlgoPostprocParams holds no params of the type expected "
                     "by this postprocessor";
      return InferErrorCode::InferInvalidInput;
    }

    FrameTransformContext prep_context{};
    if constexpr (RequiresPrepContext) {
      if (runtime_context == nullptr ||
          !runtime_context->frame_transform.has_value()) {
        LOG_ERROR_S << "Preprocessor transform context is missing from the "
                       "runtime context";
        return InferErrorCode::InferInvalidInput;
      }
      prep_context = *runtime_context->frame_transform;
    }

    return processTyped(model_output, prep_context, *params, algo_output)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }

  InferErrorCode
  batchProcess(const TensorData &model_output,
               const AlgoPostprocParams &post_args,
               std::vector<AlgoOutput> &algo_outputs,
               std::shared_ptr<RuntimeContext> &runtime_context) const final {
    if (model_output.datas.empty()) {
      LOG_ERROR_S << "model_output.datas is empty";
      return InferErrorCode::InferOutputError;
    }

    auto params = post_args.getParams<ParamsT>();
    if (params == nullptr) {
      LOG_ERROR_S << "AlgoPostprocParams holds no params of the type expected "
                     "by this postprocessor";
      return InferErrorCode::InferInvalidInput;
    }

    std::vector<FrameTransformContext> prep_contexts;
    if constexpr (RequiresPrepContext) {
      if (runtime_context == nullptr ||
          runtime_context->frame_transform_batch.empty()) {
        LOG_ERROR_S << "Batch preprocessor transform context is missing from "
                       "the runtime context";
        return InferErrorCode::InferInvalidInput;
      }
      prep_contexts = runtime_context->frame_transform_batch;
    }

    return batchProcessTyped(model_output, prep_contexts, *params,
                             algo_outputs)
               ? InferErrorCode::SUCCESS
               : InferErrorCode::InferOutputError;
  }

protected:
  virtual bool processTyped(const TensorData &, const FrameTransformContext &,
                            const ParamsT &, AlgoOutput &) const = 0;

  virtual bool batchProcessTyped(const TensorData &,
                                 const std::vector<FrameTransformContext> &,
                                 const ParamsT &,
                                 std::vector<AlgoOutput> &) const = 0;
};

} // namespace ai_core::dnn

#endif
