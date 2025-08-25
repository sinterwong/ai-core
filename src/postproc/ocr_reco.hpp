/**
 * @file ocr_reco.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-25
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFERENCE_VISION_OCR_RECO_HPP
#define AI_CORE_INFERENCE_VISION_OCR_RECO_HPP

#include "cv_generic_post_base.hpp"
namespace ai_core::dnn {
class OCRReco : public ICVGenericPostprocessor {
public:
  explicit OCRReco() {}

  virtual bool process(const TensorData &, const FramePreprocessArg &,
                       AlgoOutput &, const GenericPostParams &) const override;

private:
  std::vector<int64_t> ctcProcess(const std::vector<int64_t> &outputs) const;
};
} // namespace ai_core::dnn

#endif
