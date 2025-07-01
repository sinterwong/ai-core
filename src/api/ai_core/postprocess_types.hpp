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

#ifndef __POSTPROCESS_TYPES_HPP__
#define __POSTPROCESS_TYPES_HPP__

#include <string>
#include <vector>
namespace ai_core {

struct AnchorDetParams {
  float condThre;
  float nmsThre;
  std::vector<std::string> outputNames;
};
} // namespace ai_core

#endif // __POSTPROCESS_TYPES_HPP__