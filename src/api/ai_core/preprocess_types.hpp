/**
 * @file preprocess_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_PREPROCESS_TYPES_HPP
#define AI_CORE_PREPROCESS_TYPES_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/image_view.hpp"
#include "ai_core/typed_buffer.hpp"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace ai_core {

struct Shape {
  int w;
  int h;
  int c;
};

struct FramePreprocessArg {
  Shape model_input_shape;
  bool need_resize;
  bool is_equal_scale;
  std::vector<int> pad;

  /**
   * @brief Per-channel normalization, applied as
   *        `output[c] = (input[c] - mean_vals[c]) / std_vals[c]`.
   *
   * `std_vals` is a **divisor** (a standard deviation), not a multiplier: to
   * map 8-bit pixels into [0, 1] pass `{255, 255, 255}`, not `{1/255, ...}`.
   * Both vectors are either empty (no normalization) or sized to the model's
   * input channel count.
   */
  std::vector<float> mean_vals;
  std::vector<float> std_vals;

  /**
   * @brief Pixel format the model was trained on.
   *
   * The preprocessor converts `FrameInput::image.format` to this before
   * normalizing — it is not decorative. Leave it at the OpenCV-native default
   * for BGR-trained models; ultralytics-style models want `RGB888`.
   */
  ImagePixelFormat model_input_format = ImagePixelFormat::BGR888;

  bool hwc2chw;
  DataType data_type;
  BufferLocation output_location = BufferLocation::CPU;
  std::vector<std::string> input_names;
};

/**
 * @brief How the preprocessor mapped the source frame into the model input.
 *
 * Filled by frame preprocessors, consumed by postprocessors that report
 * coordinates in source-image space. Use the accessors rather than re-deriving
 * the scale from the raw fields — that derivation used to be copy-pasted into
 * every decoder and drifted.
 */
struct FrameTransformContext {
  bool is_equal_scale;
  Shape origin_shape;
  Shape model_input_shape;
  Rect roi; // region the preprocessor actually consumed (full frame if the
            // caller passed none)
  int top_pad = 0;
  int left_pad = 0;

  /// The region the model actually saw: the ROI when one was set, else the
  /// whole frame.
  Shape sourceShape() const noexcept {
    if (roi.area() > 0) {
      return Shape{roi.width, roi.height, origin_shape.c};
    }
    return origin_shape;
  }

  /// Model-input pixels per source pixel, as {x, y}. Equal on both axes when
  /// the preprocessor letterboxed (`is_equal_scale`).
  std::pair<float, float> scaleRatio() const noexcept {
    const Shape src = sourceShape();
    if (src.w <= 0 || src.h <= 0) {
      return {1.f, 1.f};
    }
    const float rw = static_cast<float>(model_input_shape.w) / src.w;
    const float rh = static_cast<float>(model_input_shape.h) / src.h;
    if (is_equal_scale) {
      const float r = std::min(rw, rh);
      return {r, r};
    }
    return {rw, rh};
  }

  /// Map a point from model-input space back to source-image space.
  /// Padding is zero unless the preprocessor letterboxed, so this one formula
  /// covers both the equal-scale and stretch cases.
  Point2f mapToSource(const Point2f &p) const noexcept {
    const auto [sx, sy] = scaleRatio();
    return Point2f{(p.x - left_pad) / sx + static_cast<float>(roi.x),
                   (p.y - top_pad) / sy + static_cast<float>(roi.y)};
  }

  /// Map a size (width/height) from model-input space back to source-image
  /// space. Sizes are translation-invariant: no padding, no ROI offset.
  Point2f mapSizeToSource(float w, float h) const noexcept {
    const auto [sx, sy] = scaleRatio();
    return Point2f{w / sx, h / sy};
  }
};

} // namespace ai_core

#endif