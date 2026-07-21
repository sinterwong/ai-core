/**
 * @file i_infer_engine.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-18
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFER_BASE_HPP
#define AI_CORE_INFER_BASE_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/error_code.hpp"
#include "ai_core/tensor_data.hpp"
#include <memory>

namespace ai_core::dnn {
/**
 * @brief Synchronous inference engine plugin interface.
 *
 * @par Thread-safety contract
 * A single instance must tolerate concurrent @ref infer calls (implementations
 * serialize or parallelize internally: ORT via a shared lock, NCNN/TRT via a
 * mutex). @ref initialize / @ref terminate require exclusive access. For true
 * multi-thread parallelism prefer @ref IAsyncInferEngine with per-thread
 * execution contexts.
 */
class IInferEnginePlugin {
public:
  IInferEnginePlugin() = default;

  virtual ~IInferEnginePlugin() {}

  virtual InferErrorCode initialize() = 0;

  virtual InferErrorCode infer(const TensorData &inputs,
                               TensorData &outputs) = 0;

  virtual InferErrorCode terminate() = 0;

  virtual const ModelInfo &getModelInfo() = 0;

  virtual void prettyPrintModelInfos();

protected:
  std::shared_ptr<ModelInfo> m_modelInfo;
};
} // namespace ai_core::dnn
#endif
