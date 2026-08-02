/**
 * @file postprocess_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_POSTPROCESS_TYPES_HPP
#define AI_CORE_POSTPROCESS_TYPES_HPP

#include <string>
#include <vector>
namespace ai_core {

struct GenericPostParams {
  std::vector<std::string> output_names;

  // Classification postprocessors (SoftmaxCls / ArgmaxCls) already compute the
  // whole class distribution before taking the argmax. Set this to keep it in
  // ClsRet::probs instead of throwing it away; off by default so the common
  // top-1 path stays allocation-free.
  bool keep_class_probs = false;
};

struct ConfidenceFilterParams {
  float cond_thre;
  std::vector<std::string> output_names;
};

struct AnchorDetParams {
  float cond_thre;
  float nms_thre;
  std::vector<std::string> output_names;
};
} // namespace ai_core

#endif