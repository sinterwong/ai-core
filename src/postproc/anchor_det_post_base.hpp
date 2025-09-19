/**
 * @file anchor_det_post_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __I_ANCHOR_DET_POSTPROCESSOR_HPP__
#define __I_ANCHOR_DET_POSTPROCESSOR_HPP__

#include "ai_core/algo_data_types.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core {
/**
 * @brief Interface for frame preprocessors.
 *
 */
class IAnchorDetPostprocessor {
public:
  virtual ~IAnchorDetPostprocessor() = default;

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const AnchorDetParams &, AlgoOutput &) const = 0;

  virtual bool batchProcess(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const AnchorDetParams &,
                            std::vector<AlgoOutput> &) const = 0;
};

} // namespace ai_core

#endif // __I_IMAGE_PREPROCESSOR_HPP__
