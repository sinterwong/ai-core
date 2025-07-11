/**
 * @file dnn_infer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "dnn_infer.hpp"
#include "crypto.hpp"
#include "logger.hpp"
#include <ncnn/cpu.h>
#include <ostream>
#include <stdlib.h>
#include <vector>

#ifndef _WIN32
#include <cstdlib> // For posix_memalign
#endif

namespace ai_core::dnn {

NCNNAlgoInference::~NCNNAlgoInference() { terminate(); }

InferErrorCode NCNNAlgoInference::initialize() {
  std::lock_guard lock(mtx_);

  // ensure visibility if other threads check
  isInitialized.store(false, std::memory_order_release);

  net.clear();
  blobPoolAllocator.clear();
  workspacePoolAllocator.clear();
  inputNames.clear();
  outputNames.clear();
  modelInfo.reset();
  for (void *ptr : m_aligned_buffers) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }
  m_aligned_buffers.clear();

  LOG_INFOS << "Attempting to initialize model: " << params_.name;

  try {
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    net.opt = ncnn::Option();

#if NCNN_VULKAN
    bool use_gpu = (params_.deviceType == DeviceType::GPU);
    net.opt.use_vulkan_compute = use_gpu;
    if (use_gpu) {
      LOG_INFOS << params_.name << " will attempt to load on GPU (Vulkan).";
    } else {
      LOG_INFOS << params_.name << " will load on CPU.";
    }
#else
    if (params_.deviceType == DeviceType::GPU) {
      LOG_WARNS
          << "NCNN Vulkan support is not compiled, but GPU was requested. "
             "Falling back to CPU for model: "
          << params_.name;
    } else {
      LOG_INFOS << params_.name << " will load on CPU.";
    }
#endif
    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blobPoolAllocator;
    net.opt.workspace_allocator = &workspacePoolAllocator;

    std::string paramPath = params_.modelPath + ".param";
    if (net.load_param(paramPath.c_str()) != 0) {
      LOG_ERRORS << "Failed to load model parameters: " << paramPath;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    LOG_INFOS << "Successfully loaded parameters: " << paramPath;

    std::string binPath = params_.modelPath + ".bin";

    if (params_.needDecrypt) {
      int model_load_ret = -1;
      LOG_INFOS << "Decrypting model weights: " << binPath;
      std::vector<uchar> modelData;
      std::string securityKey = SECURITY_KEY;
      auto cryptoConfig = encrypt::Crypto::deriveKeyFromCommit(securityKey);
      encrypt::Crypto crypto(cryptoConfig);
      if (!crypto.decryptData(binPath, modelData)) {
        LOG_ERRORS << "Failed to decrypt model data: " << binPath;
        return InferErrorCode::INIT_DECRYPTION_FAILED;
      }

      if (modelData.empty()) {
        LOG_ERRORS << "Decryption resulted in empty model data: " << binPath;
        return InferErrorCode::INIT_MODEL_LOAD_FAILED;
      }

      size_t dataSize = modelData.size();
      size_t alignment =
          alignof(std::max_align_t) > 4 ? alignof(std::max_align_t) : 4;
      size_t alignedSize = ncnn::alignSize(dataSize, alignment);
      void *alignedData = nullptr;

#ifdef _WIN32
      alignedData = _aligned_malloc(alignedSize, alignment);
      if (!alignedData) {
        LOG_ERRORS << "Failed to allocate aligned memory for decrypted model.";
        return InferErrorCode::INIT_MEMORY_ALLOC_FAILED;
      }
#else
      int allocRet = posix_memalign(&alignedData, alignment, alignedSize);
      if (allocRet != 0) {
        LOG_ERRORS << "posix_memalign failed with error code: " << allocRet;
        alignedData = nullptr; // Ensure pointer is null on failure
        return InferErrorCode::INIT_MEMORY_ALLOC_FAILED;
      }
#endif

      memcpy(alignedData, modelData.data(), dataSize);
      if (alignedSize > dataSize) {
        memset(static_cast<unsigned char *>(alignedData) + dataSize, 0,
               alignedSize - dataSize);
      }

      model_load_ret =
          net.load_model(static_cast<const unsigned char *>(alignedData));

      if (model_load_ret <= 0) {
        LOG_ERRORS
            << "Failed to load decrypted model weights from memory buffer.";
#ifdef _WIN32
        _aligned_free(alignedData);
#else
        free(alignedData);
#endif
        return InferErrorCode::INIT_MODEL_LOAD_FAILED;
      } else {
        m_aligned_buffers.push_back(alignedData);
        LOG_INFOS << "Successfully loaded decrypted model weights from memory.";
      }
    } else {
      if (net.load_model(binPath.c_str()) != 0) {
        LOG_ERRORS << "Failed to load model weights: " << binPath;
        return InferErrorCode::INIT_MODEL_LOAD_FAILED;
      }
      LOG_INFOS << "Successfully loaded model weights: " << binPath;
    }

    const auto &in_names = net.input_names();
    const auto &out_names = net.output_names();
    inputNames.assign(in_names.begin(), in_names.end());
    outputNames.assign(out_names.begin(), out_names.end());
    LOG_INFOS << "Successfully initialized model: " << params_.name;
    isInitialized.store(
        true, std::memory_order_release); // Set flag ONLY on full success
    return InferErrorCode::SUCCESS;

  } catch (const std::exception &e) {
    LOG_ERRORS << "Exception during initialization: " << e.what();
    isInitialized.store(false, std::memory_order_release);
    return InferErrorCode::INIT_FAILED;
  } catch (...) {
    LOG_ERRORS << "Unknown exception during initialization.";
    isInitialized.store(false, std::memory_order_release);
    return InferErrorCode::INIT_FAILED;
  }
}

InferErrorCode NCNNAlgoInference::infer(TensorData &inputs,
                                        TensorData &outputs) {
  if (!isInitialized.load(std::memory_order_acquire)) {
    LOG_ERRORS << "Inference called on uninitialized model: " << params_.name;
    return InferErrorCode::NOT_INITIALIZED;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  try {
    outputs.datas.clear();
    outputs.shapes.clear();

    if (inputs.datas.empty() && !inputNames.empty()) {
      LOG_ERRORS << "Empty input data for NCNN model: " << params_.name;
      return InferErrorCode::INFER_PREPROCESS_FAILED; // Or appropriate error
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);

    for (const auto &inputName : inputNames) {
      auto it = inputs.datas.find(inputName);
      if (it == inputs.datas.end()) {
        LOG_ERRORS << "Input tensor '" << inputName
                   << "' not found in provided inputs for NCNN model: "
                   << params_.name;
        return InferErrorCode::INFER_FAILED;
      }
      const TypedBuffer &buffer = it->second;
      if (buffer.getSizeBytes() == 0) {
        LOG_ERRORS << "Empty input data for input tensor '" << inputName;
        return InferErrorCode::INFER_FAILED;
      }
      auto shape_it = inputs.shapes.find(inputName);
      if (shape_it == inputs.shapes.end() || shape_it->second.empty()) {
        LOG_ERRORS << "Shape for input tensor '" << inputName
                   << "' not found or is empty for NCNN model: "
                   << params_.name;
        return InferErrorCode::INFER_FAILED;
      }
      const std::vector<int> &shape = shape_it->second;

      if (buffer.dataType() != DataType::FLOAT32) {
        LOG_ERRORS << "Unsupported data type for NCNN input '" << inputName
                   << "'. Expected FLOAT32.";
        return InferErrorCode::INFER_FAILED;
      }

      ncnn::Mat ncnn_in;
      if (shape.size() == 3) {
        ncnn_in = ncnn::Mat(shape[2], shape[1], shape[0],
                            const_cast<void *>(buffer.getRawHostPtr()),
                            sizeof(float));
      } else if (shape.size() == 4 && shape[0] == 1) { // NCHW, N=1
        ncnn_in = ncnn::Mat(shape[3], shape[2], shape[1],
                            const_cast<void *>(buffer.getRawHostPtr()),
                            sizeof(float));
      } else if (shape.size() == 2) { // HW
        ncnn_in = ncnn::Mat(shape[1], shape[0],
                            const_cast<void *>(buffer.getRawHostPtr()),
                            sizeof(float));
      } else if (shape.size() == 1) { // W
        ncnn_in =
            ncnn::Mat(shape[0], const_cast<void *>(buffer.getRawHostPtr()),
                      sizeof(float));
      } else {
        LOG_ERRORS << "Unsupported input shape dimension " << shape.size()
                   << " for NCNN input '" << inputName << "'.";
        return InferErrorCode::INFER_FAILED;
      }

      if (ncnn_in.empty()) {
        LOG_ERRORS << "Failed to create ncnn::Mat for input: " << inputName;
        return InferErrorCode::INFER_FAILED;
      }
      ex.input(inputName.c_str(), ncnn_in);
    }

    for (const auto &outputName : outputNames) {
      ncnn::Mat ncnn_out;
      int ret = ex.extract(outputName.c_str(), ncnn_out);
      if (ret != 0) {
        LOG_ERRORS << "Failed to extract output '" << outputName
                   << "' from NCNN model: " << params_.name;
        return InferErrorCode::INFER_FAILED;
      }

      TypedBuffer outputBuffer;
      outputBuffer.resize(ncnn_out.total());
      memcpy(outputBuffer.getRawHostPtr(), ncnn_out.data,
             outputBuffer.getSizeBytes());

      outputs.datas.insert(std::make_pair(outputName, std::move(outputBuffer)));

      std::vector<int> shapeVec;
      if (ncnn_out.dims == 1) { // Typically (Features) or (Width)
        shapeVec = {ncnn_out.w};
      } else if (ncnn_out.dims == 2) { // Typically (Height, Width)
        shapeVec = {ncnn_out.h, ncnn_out.w};
      } else if (ncnn_out.dims == 3) { // Typically (Channels, Height, Width)
        shapeVec = {ncnn_out.c, ncnn_out.h, ncnn_out.w};
      } else if (ncnn_out.dims == 4) { // (Channels, Depth, Height, Width)
        shapeVec = {ncnn_out.c, static_cast<int>(ncnn_out.elemsize), ncnn_out.h,
                    ncnn_out.w};
        // representation
        // A common interpretation for ncnn output with dims=4 is (N, C, H, W)
        // where N=1 and d is used for C if cdim is implicit.
        // For safety, let's assume it's (ncnn_out.c, ncnn_out.d, ncnn_out.h,
        // ncnn_out.w)
        // The actual interpretation depends on how the model was constructed.
        // For now, let's use a simpler representation.
        LOG_WARNINGS
            << "4D NCNN output tensor, shape interpretation might need "
               "verification.";
        // A common output format might be just a flat list of features, or
        // CHW.
        // If it's truly 4D like (1, C, H, W), then:
        // shapeVec = {1, ncnn_out.c, ncnn_out.h, ncnn_out.w}; if we assume d is
        // batch.
        // Let's stick to ncnn's perspective: c, h, w (and d if present)
        // If dims is 4, it's (w, h, depth, channels) according to some docs for
        // ncnn::Mat constructor,
        // but ncnn::Mat members are c, d, h, w.
        // Let's assume dims indicates number of dimensions, and then c,h,w (and
        // d) are populated.
        // So if dims = 4, it's (ncnn_out.c, ncnn_out.d, ncnn_out.h,
        // ncnn_out.w)
        // This is very confusing. Let's assume for vision tasks, output is
        // rarely 4D unless it's sequence.
        // Most likely, it's 3D (C,H,W) or 1D (features).
        // For now, if dims==4, we'll use {ncnn_out.c, ncnn_out.d, ncnn_out.h,
        // ncnn_out.w}
        // but this should be verified against actual model outputs.
        // A safer bet for unknown 4D is to flatten or use a known convention.
        // Let's assume it's (C,D,H,W) from ncnn if dims == 4
        shapeVec = {ncnn_out.c, ncnn_out.d, ncnn_out.h, ncnn_out.w};
      }
      outputs.shapes.insert(std::make_pair(outputName, shapeVec));
    }
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERRORS << "Std Exception during inference for " << params_.name << ": "
               << e.what();
    return InferErrorCode::INFER_FAILED;
  } catch (...) {
    LOG_ERRORS << "Unknown exception during NCNN inference for "
               << params_.name;
    return InferErrorCode::INFER_FAILED;
  }
}

const ModelInfo &NCNNAlgoInference::getModelInfo() {
  if (isInitialized.load(std::memory_order_acquire) && modelInfo) {
    std::lock_guard lock(mtx_);
    if (modelInfo) {
      return *modelInfo;
    }
  }

  {
    std::lock_guard lock(mtx_);
    if (!isInitialized.load(std::memory_order_relaxed)) {
      LOG_ERRORS << "getModelInfo called on uninitialized or terminated model.";
      static ModelInfo emptyInfo = {};
      return emptyInfo;
    }

    if (modelInfo) {
      return *modelInfo;
    }

    LOG_INFOS << "Generating model info for: " << params_.name;
    modelInfo = std::make_shared<ModelInfo>();
    modelInfo->name = params_.name;

    modelInfo->inputs.reserve(inputNames.size());
    for (const auto &inputName : inputNames) {
      modelInfo->inputs.push_back({inputName, {}});
    }

    modelInfo->outputs.reserve(outputNames.size());
    for (const auto &outputName : outputNames) {
      modelInfo->outputs.push_back({outputName, {}});
    }

    return *modelInfo;
  }
}

InferErrorCode NCNNAlgoInference::terminate() {
  std::lock_guard<std::mutex> lock(mtx_);
  LOG_INFOS << "Terminating NCNN model: " << params_.name;
  try {
    net.clear();                    // Releases network structure and weights
    blobPoolAllocator.clear();      // Clear blob memory pool
    workspacePoolAllocator.clear(); // Clear workspace memory pool

    for (void *ptr : m_aligned_buffers) {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
    m_aligned_buffers.clear();

    inputNames.clear();
    outputNames.clear();
    modelInfo.reset(); // Release shared_ptr

    LOG_INFOS << "NCNN Model terminated successfully: " << params_.name;
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERRORS << "Std Exception during termination for " << params_.name
               << ": " << e.what();
    return InferErrorCode::TERMINATE_FAILED;
  } catch (...) {
    LOG_ERRORS << "Unknown exception during NCNN model termination: "
               << params_.name;
    return InferErrorCode::TERMINATE_FAILED;
  }
}
}; // namespace ai_core::dnn
