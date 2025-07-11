/**
 * @file frame_prep.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __PREPROCESS_SINGLE_FRAME_INPUT_HPP_
#define __PREPROCESS_SINGLE_FRAME_INPUT_HPP_

#include "ai_core/algo_data_types.hpp"
#include "ai_core/tensor_data.hpp"
#include "preproc_base.hpp"

namespace ai_core::dnn {
class FramePreprocess : public PreprocssBase {
public:
  FramePreprocess() = default;
  ~FramePreprocess() = default;

  virtual bool process(AlgoInput &input, AlgoPreprocParams &params,
                       TensorData &output) override;
};
} // namespace ai_core::dnn

#endif
