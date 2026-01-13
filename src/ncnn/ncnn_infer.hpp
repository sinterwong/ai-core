/**
 * @file ncnn_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_NCNN_INFERENCE_HPP
#define AI_CORE_NCNN_INFERENCE_HPP

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "ai_core/algo_types.hpp"
#include "ai_core/i_infer_engine.hpp"
#include "ai_core/infer_config.hpp"
#include <ncnn/allocator.h>
#include <ncnn/net.h>

namespace ai_core::dnn {
class NCNNAlgoInference : public IInferEnginePlugin {
public:
  explicit NCNNAlgoInference(const AlgoConstructParams &params)
      : m_params(std::move(params.getParam<AlgoInferParams>("params"))),
        m_isInitialized(false) {
    m_blobPoolAllocator.set_size_compare_ratio(0.f);
    m_workspacePoolAllocator.set_size_compare_ratio(0.f);
  }

  virtual ~NCNNAlgoInference() override;

  virtual InferErrorCode initialize() override;

  virtual InferErrorCode infer(const TensorData &inputs,
                               TensorData &outputs) override;

  virtual const ModelInfo &getModelInfo() override;

  virtual InferErrorCode terminate() override;

protected:
  AlgoInferParams m_params;
  std::vector<std::string> m_inputNames;
  std::vector<std::string> m_outputNames;
  std::shared_ptr<ModelInfo> m_modelInfo;

  ncnn::Net m_net;
  ncnn::PoolAllocator m_blobPoolAllocator;
  ncnn::PoolAllocator m_workspacePoolAllocator;
  // For manually managed memory if needed
  std::vector<void *> m_pAlignedBuffers;

  mutable std::mutex m_mtx;
  std::atomic_bool m_isInitialized;
};
} // namespace ai_core::dnn
#endif
