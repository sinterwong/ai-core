/**
 * @file confidence_filter_post_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_I_CONFIDENCE_FILTER_POST_BASE_HPP
#define AI_CORE_I_CONFIDENCE_FILTER_POST_BASE_HPP

#include "ai_core/algo_data_types.hpp"
#include "ai_core/postproc_types.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core {
/**
 * @brief Interface for frame preprocessors.
 *
 */
class IConfidencePostprocessor {
public:
  virtual ~IConfidencePostprocessor() = default;

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const ConfidenceFilterParams &, AlgoOutput &) const = 0;
};

} // namespace ai_core

#endif // __I_IMAGE_PREPROCESSOR_HPP__
