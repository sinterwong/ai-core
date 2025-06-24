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

#include "infer_common_types.hpp"

namespace infer {

struct AnchorDetParams {
  float condThre;
  float nmsThre;
  Shape inputShape;
};
} // namespace infer

#endif // __POSTPROCESS_TYPES_HPP__