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
#include <chrono>

#include "ai_core/error_code.hpp"
#include "ai_core/logger.hpp"
#include "ai_core/tensor_data.hpp"
#include "algo_infer_impl.hpp"

namespace ai_core::dnn {
AlgoInference::Impl::Impl(const AlgoModuleTypes &algo_module_types,
                          const AlgoInferParams &infer_params)
    : m_algoModuleTypes(algo_module_types), m_inferParams(infer_params) {
  m_preprocessor =
      std::make_shared<AlgoPreproc>(m_algoModuleTypes.preproc_module);
  m_engine = std::make_shared<AlgoInferEngine>(m_algoModuleTypes.infer_module,
                                               m_inferParams);
  m_postprocessor =
      std::make_shared<AlgoPostproc>(m_algoModuleTypes.postproc_module);
};

InferErrorCode AlgoInference::Impl::initialize() {
  if (m_preprocessor->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to initialize preprocessor.";
    return InferErrorCode::InitFailed;
  }
  if (m_engine->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to initialize inference engine.";
    return InferErrorCode::InitFailed;
  }
  if (m_postprocessor->initialize() != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to initialize postprocessor.";
    return InferErrorCode::InitFailed;
  }
  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoInference::Impl::infer(
    const AlgoInput &input, const AlgoPreprocParams &preproc_params,
    const AlgoPostprocParams &postproc_params, AlgoOutput &output) {

  if (m_engine == nullptr || m_preprocessor == nullptr ||
      m_postprocessor == nullptr) {
    LOG_ERROR_S << "Please initialize first";
    return InferErrorCode::InitFailed;
  }

  std::shared_ptr<RuntimeContext> runtime_context =
      std::make_shared<RuntimeContext>();

  // prep const time
  auto start_pre = std::chrono::steady_clock::now();
  TensorData model_input;
  if (m_preprocessor->process(input, preproc_params, model_input,
                              runtime_context) != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to preprocess input.";
    return InferErrorCode::InferPreprocessFailed;
  }
  auto end_pre = std::chrono::steady_clock::now();
  auto duration_pre = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_pre - start_pre);

  // infer cost time
  auto start_infer = std::chrono::steady_clock::now();
  TensorData model_output;
  auto ret = m_engine->infer(model_input, model_output);
  if (ret != InferErrorCode::SUCCESS) {
    return ret;
  }
  auto end_infer = std::chrono::steady_clock::now();
  auto duration_infer = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_infer - start_infer);

  // post cost time
  auto start_post = std::chrono::steady_clock::now();
  if (m_postprocessor->process(model_output, postproc_params, output,
                               runtime_context) != InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to postprocess output.";
    return InferErrorCode::InferOutputError;
  }
  auto end_post = std::chrono::steady_clock::now();
  auto duration_post = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_post - start_post);

  LOG_TRACE_S << "Preprocess time: " << duration_pre.count()
              << " ms, Infer time: " << duration_infer.count()
              << " ms, Postprocess time: " << duration_post.count() << " ms.";

  return InferErrorCode::SUCCESS;
}

InferErrorCode
AlgoInference::Impl::batchInfer(const std::vector<AlgoInput> &inputs,
                                const AlgoPreprocParams &preproc_params,
                                const AlgoPostprocParams &postproc_params,
                                std::vector<AlgoOutput> &outputs) {
  if (m_engine == nullptr || m_preprocessor == nullptr ||
      m_postprocessor == nullptr) {
    LOG_ERROR_S << "Please initialize first";
    return InferErrorCode::InitFailed;
  }

  std::shared_ptr<RuntimeContext> runtime_context =
      std::make_shared<RuntimeContext>();

  // prep const time
  auto start_pre = std::chrono::steady_clock::now();
  TensorData model_input;
  if (m_preprocessor->batchProcess(inputs, preproc_params, model_input,
                                   runtime_context) !=
      InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to batch preprocess input.";
    return InferErrorCode::InferPreprocessFailed;
  }
  auto end_pre = std::chrono::steady_clock::now();
  auto duration_pre = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_pre - start_pre);

  // infer cost time
  auto start_infer = std::chrono::steady_clock::now();
  TensorData model_output;
  auto ret = m_engine->infer(model_input, model_output);
  if (ret != InferErrorCode::SUCCESS) {
    return ret;
  }
  auto end_infer = std::chrono::steady_clock::now();
  auto duration_infer = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_infer - start_infer);

  // post cost time
  auto start_post = std::chrono::steady_clock::now();
  if (m_postprocessor->batchProcess(model_output, postproc_params, outputs,
                                    runtime_context) !=
      InferErrorCode::SUCCESS) {
    LOG_ERROR_S << "Failed to batch postprocess output.";
    return InferErrorCode::InferOutputError;
  }
  auto end_post = std::chrono::steady_clock::now();
  auto duration_post = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_post - start_post);
  LOG_TRACE_S << "Batch Preprocess time: " << duration_pre.count()
              << " ms, Batch Infer time: " << duration_infer.count()
              << " ms, Batch Postprocess time: " << duration_post.count()
              << " ms.";

  return InferErrorCode::SUCCESS;
}

InferErrorCode AlgoInference::Impl::terminate() {
  m_engine->terminate();
  return InferErrorCode::SUCCESS;
}

const ModelInfo &AlgoInference::Impl::getModelInfo() const noexcept {
  if (m_engine == nullptr) {
    LOG_ERROR_S << "Please initialize first";
    static ModelInfo model_info;
    return model_info;
  }
  return m_engine->getModelInfo();
}

const AlgoModuleTypes &AlgoInference::Impl::getModuleTypes() const noexcept {
  return m_algoModuleTypes;
};

} // namespace ai_core::dnn
