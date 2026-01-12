/**
 * @file postproc_types.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-22
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef AI_CORE_POSTPROCESS_TYPES_HPP
#define AI_CORE_POSTPROCESS_TYPES_HPP

#include <string>
#include <vector>
namespace ai_core {

struct GenericPostParams {
  enum class AlgoType : int8_t {
    SoftmaxCls = 0,
    FprCls,
    RawModelOutput,
    UnetDualOutput,
    OcrReco
  };

  AlgoType algo_type;
  std::vector<std::string> output_names;
};

struct ConfidenceFilterParams {
  enum class AlgoType : int8_t { SemanticSeg = 0 };
  AlgoType algo_type;
  float cond_thre;
  std::vector<std::string> output_names;
};

struct AnchorDetParams {
  enum class AlgoType : int8_t {
    YoloDetV11 = 0,
    RtmDet,
    NanoDet,
  };

  float cond_thre;
  float nms_thre;

  AlgoType algo_type;
  std::vector<std::string> output_names;
};
} // namespace ai_core

#endif