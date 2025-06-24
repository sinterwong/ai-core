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
#include "infer_types.hpp"
#include "logger/logger.hpp"
#include <ncnn/allocator.h>
#include <ncnn/cpu.h>
#include <opencv2/core/hal/interface.h>
#include <stdlib.h>
#include <vector>

namespace infer::dnn {

InferErrorCode AlgoInference::initialize() {
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
    free(ptr);
  }
  m_aligned_buffers.clear();

  LOG_INFOS << "Attempting to initialize model: " << params->name;

  try {
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    net.opt = ncnn::Option();

#if NCNN_VULKAN
    if (!params) {
      LOG_ERRORS << "Initialization failed: params is null.";
      return InferErrorCode::INIT_FAILED;
    }
    bool use_gpu = (params->deviceType == DeviceType::GPU);
    net.opt.use_vulkan_compute = use_gpu;
    if (use_gpu) {
      LOG_INFOS << params->name << " will attempt to load on GPU (Vulkan).";
    }
#endif
    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blobPoolAllocator;
    net.opt.workspace_allocator = &workspacePoolAllocator;

    std::string paramPath = params->modelPath + ".param";
    if (net.load_param(paramPath.c_str()) != 0) {
      LOG_ERRORS << "Failed to load model parameters: " << paramPath;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    LOG_INFOS << "Successfully loaded parameters: " << paramPath;

    std::string binPath = params->modelPath + ".bin";

    if (params->needDecrypt) {
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
    LOG_INFOS << "Successfully initialized model: " << params->name;
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

InferErrorCode AlgoInference::infer(AlgoInput &input,
                                    ModelOutput &modelOutput) {
  if (!isInitialized.load(std::memory_order_acquire)) {
    LOG_ERRORS << "Inference called on uninitialized model: "
               << (params ? params->name : "Unknown");
    return InferErrorCode::NOT_INITIALIZED;
  }

  try {
    modelOutput.outputs.clear();
    modelOutput.outputShapes.clear();

    auto startPre = std::chrono::steady_clock::now();
    auto inputs = preprocess(input);
    if (inputs.empty() && !inputNames.empty()) {
      LOG_ERRORS << "Preprocessing returned empty inputs for model: "
                 << params->name;
      return InferErrorCode::INFER_PREPROCESS_FAILED;
    }
    ncnn::Extractor ex = net.create_extractor();
    for (auto const &[name, in] : inputs) {
      ex.input(name.c_str(), in);
    }
    auto endPre = std::chrono::steady_clock::now();
    auto durationPre = std::chrono::duration_cast<std::chrono::milliseconds>(
        endPre - startPre);
    LOG_INFOS << params->name << " preprocess cost " << durationPre.count()
              << " ms";

    int numOutputs = outputNames.size();
    // infer cost time
    auto start = std::chrono::steady_clock::now();
    for (auto const &output : outputNames) {
      ncnn::Mat out;
      ex.extract(output.c_str(), out);
      TypedBuffer outputData;
      outputData.dataType = DataType::FLOAT32;
      outputData.elementCount = out.total();
      auto rawData = reinterpret_cast<uint8_t *>(out.data);
      outputData.data.assign(rawData, rawData + out.total() * sizeof(float));

      std::vector<int> outputShape;
      if (out.dims == 1) {
        outputShape.push_back(out.w);
      } else if (out.dims == 2) {
        outputShape.push_back(out.h);
        outputShape.push_back(out.w);
      } else if (out.dims == 3) {
        outputShape.push_back(out.c);
        outputShape.push_back(out.h);
        outputShape.push_back(out.w);
      } else if (out.dims == 4) {
        outputShape.push_back(out.d);
        outputShape.push_back(out.c);
        outputShape.push_back(out.h);
        outputShape.push_back(out.w);
      }
      modelOutput.outputs.insert(std::make_pair(output, outputData));
      modelOutput.outputShapes.insert(std::make_pair(output, outputShape));
    }
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG_INFOS << params->name << " inference cost " << duration.count()
              << " ms";

    return InferErrorCode::SUCCESS;

  } catch (const std::exception &e) {
    LOG_ERRORS << "Exception during inference for "
               << (params ? params->name : "Unknown") << ": " << e.what();
    return InferErrorCode::INFER_FAILED;
  } catch (...) {
    LOG_ERRORS << "Unknown exception during inference for "
               << (params ? params->name : "Unknown");
    return InferErrorCode::INFER_FAILED;
  }
}

const ModelInfo &AlgoInference::getModelInfo() {
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

    if (!params) {
      LOG_ERRORS << "Cannot get model info: params is null.";
      static ModelInfo emptyInfo = {};
      return emptyInfo;
    }

    LOG_INFOS << "Generating model info for: " << params->name;
    modelInfo = std::make_shared<ModelInfo>();
    modelInfo->name = params->name;

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

InferErrorCode AlgoInference::terminate() {
  std::lock_guard lock(mtx_);
  LOG_INFOS << "Terminating model: " << (params ? params->name : "Unknown");
  try {
    net.clear();
    blobPoolAllocator.clear();
    workspacePoolAllocator.clear();

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
    modelInfo.reset();

    isInitialized.store(false, std::memory_order_release);

    LOG_INFOS << "Model terminated successfully: "
              << (params ? params->name : "Unknown");
    return InferErrorCode::SUCCESS;

  } catch (const std::exception &e) {
    LOG_ERRORS << "Exception during termination: " << e.what();
    isInitialized.store(false, std::memory_order_release);
    return InferErrorCode::TERMINATE_FAILED;
  } catch (...) {
    LOG_ERRORS << "Unknown exception during termination.";
    isInitialized.store(false, std::memory_order_release);
    return InferErrorCode::TERMINATE_FAILED;
  }
}
}; // namespace infer::dnn
