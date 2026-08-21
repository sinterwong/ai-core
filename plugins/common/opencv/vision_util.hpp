#ifndef AI_CORE_INFERENCE_VISION_UTILS_HPP
#define AI_CORE_INFERENCE_VISION_UTILS_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/image_view.hpp"
#include "ai_core/output_types.hpp"
#include "ai_core/preprocess_types.hpp"
#include <array>
#include <opencv2/core/types.hpp>

namespace ai_core::utils {

/**
 * @brief How to get from a source pixel format to the model's pixel format.
 *
 * Conversions that keep the channel count (BGR<->RGB, BGRA<->RGBA) are pure
 * permutations: they are reported via @ref channel_map so the caller can fold
 * them into a pass it is already making, at no cost. Everything else needs a
 * real cv::cvtColor.
 */
struct PixelFormatPlan {
  bool needs_cvt_color = false;
  int cvt_code = 0;
  /// Destination channel index -> source channel index. Identity when the
  /// formats match or after cvt_code has been applied.
  std::array<int, 4> channel_map{0, 1, 2, 3};

  bool isIdentity() const noexcept {
    return !needs_cvt_color && channel_map[0] == 0 && channel_map[1] == 1 &&
           channel_map[2] == 2 && channel_map[3] == 3;
  }
};

/**
 * @brief Plan the conversion from @p from to @p to.
 * @throws std::invalid_argument when the pair is not supported.
 */
PixelFormatPlan planPixelFormat(ImagePixelFormat from, ImagePixelFormat to);

/**
 * @brief Materialize the conversion from @p from to @p to.
 *
 * Returns @p src unchanged (sharing its data) when the formats already match.
 * Prefer @ref planPixelFormat when the caller can fold a channel swap into a
 * pass it is already making.
 */
cv::Mat convertPixelFormat(const cv::Mat &src, ImagePixelFormat from,
                           ImagePixelFormat to);

float calculateIoU(const BBox &bbox1, const BBox &bbox2);

std::vector<BBox> nms(const std::vector<BBox> &results, float nms_thre,
                      float conf_thre);

Shape escaleResizeWithPad(const cv::Mat &src, cv::Mat &dst, int target_width,
                          int target_height, const cv::Scalar &pad);

template <typename T>
cv::Scalar createScalarFromVector(const std::vector<T> &values) {
  cv::Scalar s;
  size_t n = std::min(values.size(), (size_t)4);
  for (size_t i = 0; i < n; ++i) {
    s[i] = static_cast<double>(values[i]);
  }
  return s;
}
} // namespace ai_core::utils
#endif
