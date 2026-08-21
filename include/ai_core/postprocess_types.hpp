
#ifndef AI_CORE_POSTPROCESS_TYPES_HPP
#define AI_CORE_POSTPROCESS_TYPES_HPP

#include <string>
#include <vector>
namespace ai_core {

/** Output selection and optional full-distribution retention. */
struct GenericPostParams {
  std::vector<std::string> output_names;

  /** Keep the computed class distribution in `ClsRet::probs`. */
  bool keep_class_probs = false;
};

/** Confidence filtering for a set of named model outputs. */
struct ConfidenceFilterParams {
  float cond_thre;
  std::vector<std::string> output_names;
};

/** Confidence and non-maximum-suppression thresholds for anchor detectors. */
struct AnchorDetParams {
  float cond_thre;
  float nms_thre;
  std::vector<std::string> output_names;
};
} // namespace ai_core

#endif
