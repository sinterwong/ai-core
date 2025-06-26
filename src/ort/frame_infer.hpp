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
#ifndef __ORT_INFERENCE_FRAME__DET_HPP_
#define __ORT_INFERENCE_FRAME__DET_HPP_

#include "dnn_infer.hpp" // Internal ORT backend header
#include "ai_core/types/algo_data_types.hpp" // For AlgoInput
#include "ai_core/types/infer_params_types.hpp" // For FrameInferParam
#include "ai_core/types/typed_buffer.hpp" // For TypedBuffer
#include <memory> // For std::unique_ptr

// cv::Mat is used in private method signatures.
// It's included via ai_core/types/algo_input_types.hpp -> opencv
namespace cv {
class Mat; // Forward declaration is good practice if full def not needed by this header directly
}

namespace ai_core::dnn {
class FrameInference : public AlgoInference {
public:
  explicit FrameInference(const FrameInferParam &param)
      : AlgoInference(param), params(std::make_unique<FrameInferParam>(param)) {
  }

private:
  std::vector<TypedBuffer> preprocess(AlgoInput &input) const override;

  std::vector<TypedBuffer> preprocessFP32(const cv::Mat &normalizedImage,
                                          int inputChannels, int inputHeight,
                                          int inputWidth) const;

  std::vector<TypedBuffer> preprocessFP16(const cv::Mat &normalizedImage,
                                          int inputChannels, int inputHeight,
                                          int inputWidth) const;

private:
  std::unique_ptr<FrameInferParam> params;
};
} // namespace ai_core::dnn
#endif