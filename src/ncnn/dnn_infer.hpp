/**
 * @file dnn_infer.hpp
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

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "ai_core/algo_data_types.hpp"
#include "ai_core/infer_params_types.hpp"
#include "infer_base.hpp"
#include <ncnn/allocator.h>
#include <ncnn/net.h>

namespace ai_core::dnn {
class NCNNAlgoInference : public InferBase {
public:
  explicit NCNNAlgoInference(const AlgoConstructParams &params)
      : mParams(std::move(params.getParam<AlgoInferParams>("params"))),
        mIsInitialized(false) {
    mBlobPoolAllocator.set_size_compare_ratio(0.f);
    mWorkspacePoolAllocator.set_size_compare_ratio(0.f);
  }

  virtual ~NCNNAlgoInference() override;

  virtual InferErrorCode initialize() override;

  virtual InferErrorCode infer(const TensorData &inputs,
                               TensorData &outputs) override;

  virtual const ModelInfo &getModelInfo() override;

  virtual InferErrorCode terminate() override;

protected:
  AlgoInferParams mParams;
  std::vector<std::string> mInputNames;
  std::vector<std::string> mOutputNames;
  std::shared_ptr<ModelInfo> mModelInfo;

  ncnn::Net mNet;
  ncnn::PoolAllocator mBlobPoolAllocator;
  ncnn::PoolAllocator mWorkspacePoolAllocator;
  // For manually managed memory if needed
  std::vector<void *> pAlignedBuffers;

  mutable std::mutex mMtx;
  std::atomic_bool mIsInitialized;
};
} // namespace ai_core::dnn
#endif
