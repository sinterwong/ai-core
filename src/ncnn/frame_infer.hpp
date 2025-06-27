/**
 * @file frame_infer_base.hpp
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

#include "ai_core/types/algo_data_types.hpp"
#include "ai_core/types/infer_params_types.hpp"
#include "dnn_infer_base.hpp"
#include <memory>

namespace ai_core::dnn {
class FrameInference : public AlgoInference {
public:
  explicit FrameInference(const FrameInferParam &param)
      : AlgoInference(param),
        params_(std::make_unique<FrameInferParam>(param)) {}

private:
  std::vector<std::pair<std::string, ncnn::Mat>>
  preprocess(AlgoInput &input) const override;

private:
  std::unique_ptr<FrameInferParam> params_;
};
} // namespace ai_core::dnn
#endif