/**
 * @file cv_generic_post_base.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __I_CV_GENERIC_POSTPROCESSOR_HPP__
#define __I_CV_GENERIC_POSTPROCESSOR_HPP__

#include "ai_core/algo_data_types.hpp"
#include "ai_core/tensor_data.hpp"

namespace ai_core {
/**
 * @brief Interface for frame preprocessors.
 *
 */
class ICVGenericPostprocessor {
public:
  virtual ~ICVGenericPostprocessor() = default;

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const GenericPostParams &, AlgoOutput &) const = 0;

  virtual bool batchProcess(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const GenericPostParams &,
                            std::vector<AlgoOutput> &) const = 0;
};

} // namespace ai_core

#endif // __I_IMAGE_PREPROCESSOR_HPP__
