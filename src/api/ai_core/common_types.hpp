/**
 * @file common_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef AI_CORE_INFER_COMMON_TYPES_HPP
#define AI_CORE_INFER_COMMON_TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace ai_core {

struct Point {
  int x{0};
  int y{0};
};

struct Point2f {
  float x{0.f};
  float y{0.f};
};

/**
 * @brief Axis-aligned rectangle. A default-constructed (empty) rect means
 * "whole frame" wherever a region-of-interest is optional.
 */
struct Rect {
  int x{0};
  int y{0};
  int width{0};
  int height{0};

  int area() const noexcept { return width * height; }
  bool empty() const noexcept { return width <= 0 || height <= 0; }

  friend bool operator==(const Rect &a, const Rect &b) noexcept {
    return a.x == b.x && a.y == b.y && a.width == b.width &&
           a.height == b.height;
  }
};

using PointList = std::vector<Point>;
using PointfList = std::vector<Point2f>;

using Contour = PointList;
using Contourf = PointfList;

enum class DeviceType { CPU = 0, GPU = 1 };

enum class DataType : uint8_t {
  FLOAT32 = 0,
  FLOAT16,
  INT32,
  INT64,
  INT8,
};

struct ModelInfo {
  std::string name;

  struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    DataType data_type;
  };

  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
};

} // namespace ai_core

#endif