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
#include <cctype>
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
  std::lock_guard lock(mMtx);

  // ensure visibility if other threads check
  mIsInitialized.store(false, std::memory_order_release);

  mNet.clear();
  mBlobPoolAllocator.clear();
  mWorkspacePoolAllocator.clear();
  mInputNames.clear();
  mOutputNames.clear();
  modelInfo.reset();
  for (void *ptr : pAlignedBuffers) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }
  pAlignedBuffers.clear();

  LOG_INFOS << "Attempting to initialize model: " << mParams.name;

  try {
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    mNet.opt = ncnn::Option();

#if NCNN_VULKAN
    bool use_gpu = (mParams.deviceType == DeviceType::GPU);
    mNet.opt.use_vulkan_compute = use_gpu;
    if (use_gpu) {
      LOG_INFOS << mParams.name << " will attempt to load on GPU (Vulkan).";
    } else {
      LOG_INFOS << mParams.name << " will load on CPU.";
    }
#else
    if (mParams.deviceType == DeviceType::GPU) {
      LOG_WARNS
          << "NCNN Vulkan support is not compiled, but GPU was requested. "
             "Falling back to CPU for model: "
          << mParams.name;
    } else {
      LOG_INFOS << mParams.name << " will load on CPU.";
    }
#endif
    mNet.opt.num_threads = ncnn::get_big_cpu_count();
    mNet.opt.blob_allocator = &mBlobPoolAllocator;
    mNet.opt.workspace_allocator = &mWorkspacePoolAllocator;

    std::string paramPath = mParams.modelPath + ".param";
    if (mNet.load_param(paramPath.c_str()) != 0) {
      LOG_ERRORS << "Failed to load model parameters: " << paramPath;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    LOG_INFOS << "Successfully loaded parameters: " << paramPath;

    std::string binPath = mParams.modelPath + ".bin";

    if (mParams.needDecrypt) {
      int model_load_ret = -1;
      LOG_INFOS << "Decrypting model weights: " << binPath;
      std::vector<u_char> modelData;
      auto cryptoConfig =
          encrypt::Crypto::deriveKeyFromCommit(mParams.decryptkeyStr);
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
          mNet.load_model(static_cast<const unsigned char *>(alignedData));

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
        pAlignedBuffers.push_back(alignedData);
        LOG_INFOS << "Successfully loaded decrypted model weights from memory.";
      }
    } else {
      if (mNet.load_model(binPath.c_str()) != 0) {
        LOG_ERRORS << "Failed to load model weights: " << binPath;
        return InferErrorCode::INIT_MODEL_LOAD_FAILED;
      }
      LOG_INFOS << "Successfully loaded model weights: " << binPath;
    }

    const auto &in_names = mNet.input_names();
    const auto &out_names = mNet.output_names();
    mInputNames.assign(in_names.begin(), in_names.end());
    mOutputNames.assign(out_names.begin(), out_names.end());
    LOG_INFOS << "Successfully initialized model: " << mParams.name;
    mIsInitialized.store(
        true, std::memory_order_release); // Set flag ONLY on full success
    return InferErrorCode::SUCCESS;

  } catch (const std::exception &e) {
    LOG_ERRORS << "Exception during initialization: " << e.what();
    mIsInitialized.store(false, std::memory_order_release);
    return InferErrorCode::INIT_FAILED;
  } catch (...) {
    LOG_ERRORS << "Unknown exception during initialization.";
    mIsInitialized.store(false, std::memory_order_release);
    return InferErrorCode::INIT_FAILED;
  }
}

InferErrorCode NCNNAlgoInference::infer(const TensorData &inputs,
                                        TensorData &outputs) {
  if (!mIsInitialized.load(std::memory_order_acquire)) {
    LOG_ERRORS << "Inference called on uninitialized model: " << mParams.name;
    return InferErrorCode::NOT_INITIALIZED;
  }

  std::lock_guard<std::mutex> lock(mMtx);

  try {
    outputs.datas.clear();
    outputs.shapes.clear();

    if (inputs.datas.empty() && !mInputNames.empty()) {
      LOG_ERRORS << "Empty input data for NCNN model: " << mParams.name;
      return InferErrorCode::INFER_PREPROCESS_FAILED; // Or appropriate error
    }

    ncnn::Extractor ex = mNet.create_extractor();
    ex.set_light_mode(true);

    for (const auto &inputName : mInputNames) {
      auto it = inputs.datas.find(inputName);
      if (it == inputs.datas.end()) {
        LOG_ERRORS << "Input tensor '" << inputName
                   << "' not found in provided inputs for NCNN model: "
                   << mParams.name;
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
                   << mParams.name;
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

    for (const auto &outputName : mOutputNames) {
      ncnn::Mat ncnn_out;
      int ret = ex.extract(outputName.c_str(), ncnn_out);
      if (ret != 0) {
        LOG_ERRORS << "Failed to extract output '" << outputName
                   << "' from NCNN model: " << mParams.name;
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
    LOG_ERRORS << "Std Exception during inference for " << mParams.name << ": "
               << e.what();
    return InferErrorCode::INFER_FAILED;
  } catch (...) {
    LOG_ERRORS << "Unknown exception during NCNN inference for "
               << mParams.name;
    return InferErrorCode::INFER_FAILED;
  }
}

const ModelInfo &NCNNAlgoInference::getModelInfo() {
  if (mIsInitialized.load(std::memory_order_acquire) && modelInfo) {
    std::lock_guard lock(mMtx);
    if (modelInfo) {
      return *modelInfo;
    }
  }

  {
    std::lock_guard lock(mMtx);
    if (!mIsInitialized.load(std::memory_order_relaxed)) {
      LOG_ERRORS << "getModelInfo called on uninitialized or terminated model.";
      static ModelInfo emptyInfo = {};
      return emptyInfo;
    }

    if (modelInfo) {
      return *modelInfo;
    }

    LOG_INFOS << "Generating model info for: " << mParams.name;
    modelInfo = std::make_shared<ModelInfo>();
    modelInfo->name = mParams.name;

    modelInfo->inputs.reserve(mInputNames.size());
    for (const auto &inputName : mInputNames) {
      modelInfo->inputs.push_back({inputName, {}});
    }

    modelInfo->outputs.reserve(mOutputNames.size());
    for (const auto &outputName : mOutputNames) {
      modelInfo->outputs.push_back({outputName, {}});
    }

    return *modelInfo;
  }
}

InferErrorCode NCNNAlgoInference::terminate() {
  std::lock_guard<std::mutex> lock(mMtx);
  LOG_INFOS << "Terminating NCNN model: " << mParams.name;
  try {
    mNet.clear();                    // Releases network structure and weights
    mBlobPoolAllocator.clear();      // Clear blob memory pool
    mWorkspacePoolAllocator.clear(); // Clear workspace memory pool

    for (void *ptr : pAlignedBuffers) {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
    pAlignedBuffers.clear();

    mInputNames.clear();
    mOutputNames.clear();
    modelInfo.reset(); // Release shared_ptr

    LOG_INFOS << "NCNN Model terminated successfully: " << mParams.name;
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERRORS << "Std Exception during termination for " << mParams.name
               << ": " << e.what();
    return InferErrorCode::TERMINATE_FAILED;
  } catch (...) {
    LOG_ERRORS << "Unknown exception during NCNN model termination: "
               << mParams.name;
    return InferErrorCode::TERMINATE_FAILED;
  }
}
}; // namespace ai_core::dnn
