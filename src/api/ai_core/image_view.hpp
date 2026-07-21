/**
 * @file image_view.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief Non-owning image view: the public API's image currency.
 * @version 0.1
 * @date 2026-07-17
 *
 * @copyright Copyright (c) 2026
 *
 */
#ifndef AI_CORE_IMAGE_VIEW_HPP
#define AI_CORE_IMAGE_VIEW_HPP

#include <cstddef>
#include <cstdint>

namespace ai_core {

/**
 * @brief Pixel layout of an 8-bit image. Channel order is part of the format.
 */
enum class ImagePixelFormat : uint8_t {
  GRAY8 = 0,
  BGR888,
  RGB888,
  BGRA8888,
  RGBA8888,
};

constexpr int channelCount(ImagePixelFormat format) noexcept {
  switch (format) {
  case ImagePixelFormat::GRAY8:
    return 1;
  case ImagePixelFormat::BGR888:
  case ImagePixelFormat::RGB888:
    return 3;
  case ImagePixelFormat::BGRA8888:
  case ImagePixelFormat::RGBA8888:
    return 4;
  }
  return 0;
}

/**
 * @brief Non-owning view over interleaved 8-bit image data.
 *
 * The caller keeps the pixel buffer alive for the duration of any call that
 * receives the view (like std::string_view). Rows are `stride` bytes apart;
 * a stride of 0 means tightly packed (width * channels).
 */
struct ImageView {
  const uint8_t *data{nullptr};
  int width{0};
  int height{0};
  size_t stride{0};
  ImagePixelFormat format{ImagePixelFormat::BGR888};

  int channels() const noexcept { return channelCount(format); }

  size_t strideBytes() const noexcept {
    return stride != 0 ? stride : static_cast<size_t>(width) * channels();
  }

  bool empty() const noexcept {
    return data == nullptr || width <= 0 || height <= 0;
  }
};

} // namespace ai_core

#endif // AI_CORE_IMAGE_VIEW_HPP
