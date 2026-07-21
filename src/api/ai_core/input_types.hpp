/**
 * @file input_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_ALGO_INPUT_TYPES_HPP
#define AI_CORE_ALGO_INPUT_TYPES_HPP

#include <optional>
#include <vector>

#include "ai_core/common_types.hpp"
#include "ai_core/image_view.hpp"

namespace ai_core {

/**
 * @brief One frame handed to the pipeline. The pixel buffer behind `image`
 * must stay alive until the infer call returns. No `roi` means "whole frame".
 */
struct FrameInput {
  ImageView image;
  std::optional<Rect> roi;
};

struct FrameInputWithMask {
  FrameInput frame_input;
  std::vector<Rect> mask_regions;
};

} // namespace ai_core

#endif