/**
 * @file frame_with_mask_prep.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __PREPROCESS_SINGLE_FRAME_WITH_MASK_INPUT_HPP_
#define __PREPROCESS_SINGLE_FRAME_WITH_MASK_INPUT_HPP_

#include "ai_core/algo_data_types.hpp"
#include "ai_core/preproc_base.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core::dnn {
class FrameWithMaskPreprocess : public IPreprocssPlugin {
public:
  FrameWithMaskPreprocess() = default;
  ~FrameWithMaskPreprocess() = default;

  virtual bool process(const AlgoInput &, const AlgoPreprocParams &,
                       TensorData &,
                       std::shared_ptr<RuntimeContext> &) const override;

  virtual bool batchProcess(const std::vector<AlgoInput> &,
                            const AlgoPreprocParams &, TensorData &,
                            std::shared_ptr<RuntimeContext> &) const override;
};
} // namespace ai_core::dnn

#endif
