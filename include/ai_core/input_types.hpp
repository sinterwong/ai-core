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
  std::optional<Rect>
      roi; ///< Source-image ROI; no value selects the full frame.
};

/** Frame input plus source-image rectangles excluded by mask-aware plugins. */
struct FrameInputWithMask {
  FrameInput frame_input;
  std::vector<Rect> mask_regions;
};

} // namespace ai_core

#endif
