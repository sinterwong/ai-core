/**
 * @file raw_feature.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-02-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_RAW_FEATURE_HPP_
#define __INFERENCE_VISION_RAW_FEATURE_HPP_

#include "cv_generic_post_base.hpp"
namespace ai_core::dnn {
class RawFeature : public ICVGenericPostprocessor {
public:
  explicit RawFeature() {}

  virtual bool process(const TensorData &, const FrameTransformContext &,
                       const GenericPostParams &, AlgoOutput &) const override;

  virtual bool batchProcess(const TensorData &,
                            const std::vector<FrameTransformContext> &,
                            const GenericPostParams &,
                            std::vector<AlgoOutput> &) const override;

private:
  FeatureRet processSingleItem(const float *featureData, int numFeatures) const;
};
} // namespace ai_core::dnn

#endif
