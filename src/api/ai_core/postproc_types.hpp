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

#ifndef __POSTPROCESS_TYPES_HPP__
#define __POSTPROCESS_TYPES_HPP__

#include <string>
#include <vector>
namespace ai_core {

struct GenericPostParams {
  enum class AlogType : int8_t {
    SOFTMAX_CLS = 0,
    FPR_CLS,
    FPR_FEAT,
    UNET_DUAL_OUTPUT
  };

  AlogType algoType;
  std::vector<std::string> outputNames;
};

struct ConfidenceFilterParams {
  enum class AlgoType : int8_t { SEMANTIC_SEG = 0 };
  AlgoType algoType;
  float condThre;
  std::vector<std::string> outputNames;
};

struct AnchorDetParams {
  enum class AlogType : int8_t {
    YOLO_DET_V11 = 0,
    RTM_DET,
    NANO_DET,
  };

  float condThre;
  float nmsThre;

  AlogType algoType;
  std::vector<std::string> outputNames;
};
} // namespace ai_core

#endif // __POSTPROCESS_TYPES_HPP__