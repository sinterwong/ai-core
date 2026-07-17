/**
 * @file postproc_types.hpp
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