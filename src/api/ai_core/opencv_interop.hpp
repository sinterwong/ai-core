/**
 * @file opencv_interop.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Zero-cost conversions between ai_core value types and OpenCV.
 *
 * This is the only public header that includes OpenCV. It is opt-in: nothing
 * else in api/ai_core depends on it.
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_OPENCV_INTEROP_HPP
#define AI_CORE_OPENCV_INTEROP_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/image_view.hpp"

#include <opencv2/core.hpp>
#include <stdexcept>

namespace ai_core::interop {

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

inline cv::Rect toCv(const Rect &r) noexcept {
  return {r.x, r.y, r.width, r.height};
}
inline Rect fromCv(const cv::Rect &r) noexcept {
  return {r.x, r.y, r.width, r.height};
}

inline cv::Point toCv(const Point &p) noexcept { return {p.x, p.y}; }
inline Point fromCv(const cv::Point &p) noexcept { return {p.x, p.y}; }

inline cv::Point2f toCv(const Point2f &p) noexcept { return {p.x, p.y}; }
inline Point2f fromCv(const cv::Point2f &p) noexcept { return {p.x, p.y}; }

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

/**
 * @brief Wrap a cv::Mat as an ImageView (zero copy).
 *
 * The Mat must be 8-bit with 1, 3 or 4 channels and must outlive the view.
 * The pixel format cannot be derived from a Mat (OpenCV convention is BGR),
 * so it defaults per channel count: 1 -> GRAY8, 3 -> BGR888, 4 -> BGRA8888.
 * Pass @p format explicitly when the data is RGB/RGBA.
 */
inline ImageView viewFromMat(const cv::Mat &mat) {
  if (mat.empty()) {
    return {};
  }
  if (mat.depth() != CV_8U) {
    throw std::invalid_argument(
        "ImageView interop requires an 8-bit (CV_8U) Mat.");
  }
  ImagePixelFormat format;
  switch (mat.channels()) {
  case 1:
    format = ImagePixelFormat::GRAY8;
    break;
  case 3:
    format = ImagePixelFormat::BGR888;
    break;
  case 4:
    format = ImagePixelFormat::BGRA8888;
    break;
  default:
    throw std::invalid_argument(
        "ImageView interop supports 1/3/4 channel Mats.");
  }
  return {mat.data, mat.cols, mat.rows, mat.step, format};
}

inline ImageView viewFromMat(const cv::Mat &mat, ImagePixelFormat format) {
  ImageView view = viewFromMat(mat);
  if (!view.empty() && channelCount(format) != mat.channels()) {
    throw std::invalid_argument(
        "ImagePixelFormat channel count does not match the Mat.");
  }
  view.format = format;
  return view;
}

/**
 * @brief Wrap an ImageView as a cv::Mat header (zero copy).
 *
 * The view's buffer must outlive the returned Mat. The Mat header does not
 * carry channel-order information; callers needing RGB handling must consult
 * view.format themselves.
 */
inline cv::Mat matFromView(const ImageView &view) {
  if (view.empty()) {
    return {};
  }
  return cv::Mat(view.height, view.width, CV_8UC(view.channels()),
                 const_cast<uint8_t *>(view.data), view.strideBytes());
}

} // namespace ai_core::interop

#endif // AI_CORE_OPENCV_INTEROP_HPP
