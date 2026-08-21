
#ifndef AI_CORE_ALGO_OUTPUT_TYPES_HPP
#define AI_CORE_ALGO_OUTPUT_TYPES_HPP

#include "ai_core/common_types.hpp"
#include "ai_core/tensor_data.hpp"
#include <map>
#include <memory>
#include <vector>

namespace ai_core {
/** One detection in source-image coordinates. */
struct BBox {
  Rect rect;   ///< Axis-aligned bounding box in source-image pixels.
  float score; ///< Model/postprocessor confidence.
  int label;   ///< Zero-based class index.
};

/**
 * @brief Top-1 classification result.
 *
 * `probs` carries the full class distribution the postprocessor computed on
 * the way to the argmax. It is empty unless
 * `GenericPostParams::keep_class_probs` is set — sliding-window / voting
 * consumers need the distribution, everyone else should not pay for the
 * allocation.
 */
struct ClsRet {
  float score;
  int label;
  std::vector<float> probs;
};

using RawModelOutput = TensorData;

/** Classification result with an additional BI-RADS category. */
struct FprClsRet {
  float score;
  int label;
  int birad;
  std::vector<float> score_probs;
};

/** Object-detection results for one input image. */
struct DetRet {
  std::vector<BBox> bboxes;
};

/** Segmentation contours grouped by zero-based class index. */
struct SegRet {
  std::map<int, std::vector<Contour>> cls_to_contours;
};

/**
 * @brief Raw dual-headed segmentation output. `mask` is an INT8/UINT8 class
 * map and `prob` a FLOAT32 probability map, both shaped `{height, width}`.
 */
struct DualRawSegRet {
  Tensor mask;
  Tensor prob;
  Rect roi;
  float ratio;
  int left_shift;
  int top_shift;
};

/** CTC-style OCR output sequence and its valid length. */
struct OCRRecoRet {
  int64_t output_lengths;
  std::vector<int64_t> outputs;
};

} // namespace ai_core

#endif
