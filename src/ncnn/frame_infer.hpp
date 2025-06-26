/**
 * @file frame_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-19
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __NCNN_INFERENCE_FRAME__DET_HPP_
#define __NCNN_INFERENCE_FRAME__DET_HPP_

#include "dnn_infer.hpp" // Internal NCNN backend header
#include "ai_core/types/algo_data_types.hpp" // For AlgoInput
#include "ai_core/types/infer_params_types.hpp" // For FrameInferParam
#include <memory> // For std::unique_ptr

// Forward declare ncnn::Mat if its full definition is not needed here
// Or ensure "ncnn/mat.h" (or similar) is included by dnn_infer.hpp
namespace ncnn {
class Mat;
}

namespace ai_core::dnn {
class FrameInference : public AlgoInference {
public:
  explicit FrameInference(const FrameInferParam &param)
      : AlgoInference(param), params(std::make_unique<FrameInferParam>(param)) {
  }

private:
  std::vector<std::pair<std::string, ncnn::Mat>>
  preprocess(AlgoInput &input) const override;

private:
  std::unique_ptr<FrameInferParam> params;
};
} // namespace ai_core::dnn
#endif