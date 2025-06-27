/**
 * @file dnn_infer_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __NCNN_INFERENCE_HPP_
#define __NCNN_INFERENCE_HPP_

#include "ai_core/types/algo_data_types.hpp"
#include "ai_core/types/infer_params_types.hpp"
#include "infer_base.hpp"
#include <atomic>
#include <mutex>
#include <ncnn/net.h>

namespace ai_core::dnn {
class NCNNAlgoInference : public InferBase {
public:
  NCNNAlgoInference(const AlgoConstructParams &params)
      : params_(std::move(params.getParam<AlgoInferParams>("params"))),
        isInitialized(false) {
    blobPoolAllocator.set_size_compare_ratio(0.f);
    workspacePoolAllocator.set_size_compare_ratio(0.f);
  }

  virtual ~NCNNAlgoInference() override {
    net.clear();
    blobPoolAllocator.clear();
    workspacePoolAllocator.clear();
    for (void *ptr : m_aligned_buffers) {
      free(ptr);
    }
    m_aligned_buffers.clear();
    inputNames.clear();
    outputNames.clear();
  }

  virtual InferErrorCode initialize() override;

  virtual InferErrorCode infer(TensorData &inputs,
                               TensorData &outputs) override;

  virtual const ModelInfo &getModelInfo() override;

  virtual InferErrorCode terminate() override;

protected:
  virtual std::vector<std::pair<std::string, ncnn::Mat>>
  preprocess(AlgoInput &input) const = 0;

protected:
  AlgoInferParams params_;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;

  ncnn::Net net;

  ncnn::PoolAllocator blobPoolAllocator;
  ncnn::PoolAllocator workspacePoolAllocator;

private:
  std::vector<void *> m_aligned_buffers;
  mutable std::mutex mtx_;
  std::atomic_bool isInitialized;
};
} // namespace ai_core::dnn
#endif
