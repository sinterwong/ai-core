/**
 * @file frame_prep.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __PREPROCESS_SINGLE_FRAME_INPUT_HPP_
#define __PREPROCESS_SINGLE_FRAME_INPUT_HPP_

#include "ai_core/types/algo_data_types.hpp"
#include "ai_core/types/tensor_data.hpp"
#include "preproc_base.hpp"
#include <memory>

#ifdef WITH_NCNN
#include "ncnn_image_preprocessor.hpp"
using ai_core::dnn::ncnn::ImagePreprocessor;
#else
#include "cpu_image_preprocessor.hpp"
using ai_core::dnn::cpu::ImagePreprocessor;
#endif

namespace ai_core::dnn {
class FramePreprocess : public PreprocssBase {
public:
  FramePreprocess();

  virtual bool process(AlgoInput &input, AlgoPreprocParams &params,
                       TensorData &output) override;

private:
  std::unique_ptr<ImagePreprocessor> processor_;
};
} // namespace ai_core::dnn

#endif
