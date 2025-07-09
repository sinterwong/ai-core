/**
 * @file dnn_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-09
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <thread>

#include "crypto.hpp"
#include "dnn_infer.hpp"
#include "logger.hpp"

#include <NvInfer.h>

namespace ai_core::dnn {

TrtAlgoInference::TrtAlgoInference(const AlgoConstructParams &params)
    : params_(std::move(params.getParam<AlgoInferParams>("params"))) {}

InferErrorCode TrtAlgoInference::initialize() {}

InferErrorCode TrtAlgoInference::infer(TensorData &inputs,
                                       TensorData &outputs) {}

InferErrorCode TrtAlgoInference::terminate() {}

const ModelInfo &TrtAlgoInference::getModelInfo() {}
}; // namespace ai_core::dnn
