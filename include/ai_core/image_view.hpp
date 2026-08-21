#ifndef AI_CORE_IMAGE_VIEW_HPP
#define AI_CORE_IMAGE_VIEW_HPP

#include <cstddef>
#include <cstdint>

namespace ai_core {

/**
 * @brief Channel count and byte order for one interleaved 8-bit pixel.
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
 * The caller owns the pixel buffer and keeps it alive while the view is in
 * use. Rows are `stride` bytes apart; zero means tightly packed
 * (`width * channels()`). Width and height are measured in pixels.
 */
struct ImageView {
  const uint8_t *data{nullptr};
  int width{0};
  int height{0};
  size_t stride{0}; ///< Row stride in bytes; zero selects the packed stride.
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
