/**
 * @file infer_common_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFER_COMMON_TYPES_HPP__
#define __INFER_COMMON_TYPES_HPP__

#include <string>
#include <vector>

namespace cv {
template <typename _Tp> class Rect_;
using Rect = Rect_<int>;

class Mat;

template <typename _Tp> class Point_;
using Point = Point_<int>;
using Point2f = Point_<float>;
} // namespace cv
namespace ai_core {

using PointList = std::vector<cv::Point>;
using PointfList = std::vector<cv::Point2f>;

using Contour = PointList;
using Contourf = PointfList;

enum class DeviceType { CPU = 0, GPU = 1 };

struct Shape {
  int w;
  int h;
  int c;
};

enum class DataType : u_char {
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
    DataType dataType;
  };

  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
};

} // namespace ai_core

#endif // __INFER_COMMON_TYPES_HPP__