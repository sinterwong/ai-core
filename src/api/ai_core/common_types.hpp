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

namespace cv {
template <typename Tp> class Rect_;
using Rect = Rect_<int>;

class Mat;

template <typename Tp> class Point_;
using Point = Point_<int>;
using Point2f = Point_<float>;
} // namespace cv
namespace ai_core {

using PointList = std::vector<cv::Point>;
using PointfList = std::vector<cv::Point2f>;

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