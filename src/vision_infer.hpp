/**
 * @file vision_infer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-24
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef __INFERENCE_VISION_INFER_HPP__
#define __INFERENCE_VISION_INFER_HPP__

#include "ai_core/algo_infer_base.hpp"
#include "ai_core/types/algo_data_types.hpp"
#include "ai_core/types/infer_common_types.hpp"
#include "ai_core/types/infer_error_code.hpp"
#include "frame_infer.hpp"
#include "vision.hpp"
#include <memory>

namespace ai_core::dnn::vision {
class VisionInfer : public AlgoInferBase {
public:
  VisionInfer(const std::string &moduleName, const AlgoInferParams &param,
              const AlgoPostprocParams &postproc);

  virtual InferErrorCode initialize() override;

  virtual InferErrorCode infer(AlgoInput &input, AlgoOutput &output) override;

  virtual InferErrorCode terminate() override;

  const ModelInfo &getModelInfo() const noexcept override;

  virtual const std::string &getModuleName() const noexcept override;

private:
  std::string moduleName;
  AlgoInferParams inferParams;
  AlgoPostprocParams postprocParams;

  std::shared_ptr<FrameInference> engine;
  std::shared_ptr<VisionBase> vision;
};
} // namespace ai_core::dnn::vision
#endif
