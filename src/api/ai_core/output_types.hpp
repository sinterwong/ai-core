/**
 * @file output_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_ALGO_OUTPUT_TYPES_HPP
#define AI_CORE_ALGO_OUTPUT_TYPES_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/tensor_data.hpp"
#include <map>
#include <memory>
#include <vector>

namespace ai_core {
struct BBox {
  Rect rect;
  float score;
  int label;
};

struct ClsRet {
  float score;
  int label;
};

using RawModelOutput = TensorData;

struct FprClsRet {
  float score;
  int label;
  int birad;
  std::vector<float> score_probs;
};

struct DetRet {
  std::vector<BBox> bboxes;
};

struct SegRet {
  std::map<int, std::vector<Contour>> cls_to_contours;
};

/**
 * @brief Raw dual-headed segmentation output. `mask` is an INT8/UINT8 class
 * map and `prob` a FLOAT32 probability map, both shaped {h, w}.
 */
struct DualRawSegRet {
  Tensor mask;
  Tensor prob;
  Rect roi;
  float ratio;
  int left_shift;
  int top_shift;
};

struct OCRRecoRet {
  int64_t output_lengths;
  std::vector<int64_t> outputs;
};

} // namespace ai_core

#endif