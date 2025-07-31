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
#include "ai_core/tensor_data.hpp"
#include "preproc_base.hpp"

namespace ai_core::dnn {
class FrameWithMaskPreprocess : public PreprocssBase {
public:
  FrameWithMaskPreprocess() = default;
  ~FrameWithMaskPreprocess() = default;

  virtual bool process(AlgoInput &input, AlgoPreprocParams &params,
                       TensorData &output) const override;
};
} // namespace ai_core::dnn

#endif
