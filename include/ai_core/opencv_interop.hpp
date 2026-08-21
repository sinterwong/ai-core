#ifndef AI_CORE_OPENCV_INTEROP_HPP
#define AI_CORE_OPENCV_INTEROP_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/image_view.hpp"

#include <opencv2/core.hpp>
#include <stdexcept>

namespace ai_core::interop {

// Geometry conversions preserve coordinates and extents.

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

/**
 * @brief Wrap a cv::Mat as an ImageView (zero copy).
 *
 * The `cv::Mat` must be 8-bit with 1, 3, or 4 channels and must outlive the
 * view. Channel order is not encoded by `cv::Mat`, so the default is `GRAY8`,
 * `BGR888`, or `BGRA8888` according to channel count. Use the overload with an
 * explicit format for RGB or RGBA data.
 *
 * @throws std::invalid_argument For unsupported depth or channel count.
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
 * carry channel-order information; callers must preserve `view.format`
 * separately when that distinction matters.
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
