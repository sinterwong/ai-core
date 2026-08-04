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

#include "ncnn_infer.hpp"
#include "ai_core/logger.hpp"
#include "crypto.hpp"
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
  std::lock_guard lock(m_mtx);

  // ensure visibility if other threads check
  m_isInitialized.store(false, std::memory_order_release);

  m_net.clear();
  m_blobPoolAllocator.clear();
  m_workspacePoolAllocator.clear();
  m_inputNames.clear();
  m_outputNames.clear();
  m_modelInfo.reset();
  for (void *ptr : m_pAlignedBuffers) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }
  m_pAlignedBuffers.clear();

  LOG_INFO_S << "Attempting to initialize model: " << m_params.name;

  try {
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    m_net.opt = ncnn::Option();

#if NCNN_VULKAN
    bool use_gpu = (m_params.device_type == DeviceType::GPU);
    m_net.opt.use_vulkan_compute = use_gpu;
    if (use_gpu) {
      LOG_INFO_S << m_params.name << " will attempt to load on GPU (Vulkan).";
    } else {
      LOG_INFO_S << m_params.name << " will load on CPU.";
    }
#else
    if (mParams.deviceType == DeviceType::GPU) {
      LOG_WARNS
          << "NCNN Vulkan support is not compiled, but GPU was requested. "
             "Falling back to CPU for model: "
          << mParams.name;
    } else {
      LOG_INFO_S << mParams.name << " will load on CPU.";
    }
#endif
    m_net.opt.num_threads = ncnn::get_big_cpu_count();
    m_net.opt.blob_allocator = &m_blobPoolAllocator;
    m_net.opt.workspace_allocator = &m_workspacePoolAllocator;

    std::string param_path = m_params.model_path + ".param";
    if (m_net.load_param(param_path.c_str()) != 0) {
      LOG_ERROR_S << "Failed to load model parameters: " << param_path;
      return InferErrorCode::InitModelLoadFailed;
    }
    LOG_INFO_S << "Successfully loaded parameters: " << param_path;

    std::string bin_path = m_params.model_path + ".bin";

    if (m_params.need_decrypt) {
      int model_load_ret = -1;
      LOG_INFO_S << "Decrypting model weights: " << bin_path;
      std::vector<uint8_t> model_data;
      auto crypto_config =
          encrypt::Crypto::deriveKeyFromCommit(m_params.decryptkey_str);
      encrypt::Crypto crypto(crypto_config);
      if (!crypto.decryptData(bin_path, model_data)) {
        LOG_ERROR_S << "Failed to decrypt model data: " << bin_path;
        return InferErrorCode::InitDecryptionFailed;
      }

      if (model_data.empty()) {
        LOG_ERROR_S << "Decryption resulted in empty model data: " << bin_path;
        return InferErrorCode::InitModelLoadFailed;
      }

      size_t data_size = model_data.size();
      size_t alignment =
          alignof(std::max_align_t) > 4 ? alignof(std::max_align_t) : 4;
      size_t aligned_size = ncnn::alignSize(data_size, alignment);
      void *aligned_data = nullptr;

#ifdef _WIN32
      alignedData = _aligned_malloc(alignedSize, alignment);
      if (!alignedData) {
        LOG_ERROR_S << "Failed to allocate aligned memory for decrypted model.";
        return InferErrorCode::INIT_MEMORY_ALLOC_FAILED;
      }
#else
      int alloc_ret = posix_memalign(&aligned_data, alignment, aligned_size);
      if (alloc_ret != 0) {
        LOG_ERROR_S << "posix_memalign failed with error code: " << alloc_ret;
        aligned_data = nullptr; // Ensure pointer is null on failure
        return InferErrorCode::InitMemoryAllocFailed;
      }
#endif

      memcpy(aligned_data, model_data.data(), data_size);
      if (aligned_size > data_size) {
        memset(static_cast<unsigned char *>(aligned_data) + data_size, 0,
               aligned_size - data_size);
      }

      model_load_ret =
          m_net.load_model(static_cast<const unsigned char *>(aligned_data));

      if (model_load_ret <= 0) {
        LOG_ERROR_S
            << "Failed to load decrypted model weights from memory buffer.";
#ifdef _WIN32
        _aligned_free(alignedData);
#else
        free(aligned_data);
#endif
        return InferErrorCode::InitModelLoadFailed;
      } else {
        m_pAlignedBuffers.push_back(aligned_data);
        LOG_INFO_S
            << "Successfully loaded decrypted model weights from memory.";
      }
    } else {
      if (m_net.load_model(bin_path.c_str()) != 0) {
        LOG_ERROR_S << "Failed to load model weights: " << bin_path;
        return InferErrorCode::InitModelLoadFailed;
      }
      LOG_INFO_S << "Successfully loaded model weights: " << bin_path;
    }

    const auto &in_names = m_net.input_names();
    const auto &out_names = m_net.output_names();
    m_inputNames.assign(in_names.begin(), in_names.end());
    m_outputNames.assign(out_names.begin(), out_names.end());
    LOG_INFO_S << "Successfully initialized model: " << m_params.name;
    m_isInitialized.store(
        true, std::memory_order_release); // Set flag ONLY on full success
    return InferErrorCode::SUCCESS;

  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception during initialization: " << e.what();
    m_isInitialized.store(false, std::memory_order_release);
    return InferErrorCode::InitFailed;
  } catch (...) {
    LOG_ERROR_S << "Unknown exception during initialization.";
    m_isInitialized.store(false, std::memory_order_release);
    return InferErrorCode::InitFailed;
  }
}

InferErrorCode NCNNAlgoInference::infer(const TensorData &inputs,
                                        TensorData &outputs) {
  if (!m_isInitialized.load(std::memory_order_acquire)) {
    LOG_ERROR_S << "Inference called on uninitialized model: " << m_params.name;
    return InferErrorCode::NotInitialized;
  }

  std::lock_guard<std::mutex> lock(m_mtx);

  try {
    outputs.clear();

    if (inputs.empty() && !m_inputNames.empty()) {
      LOG_ERROR_S << "Empty input data for NCNN model: " << m_params.name;
      return InferErrorCode::InferPreprocessFailed; // Or appropriate error
    }

    ncnn::Extractor ex = m_net.create_extractor();
    ex.set_light_mode(true);

    for (const auto &input_name : m_inputNames) {
      const Tensor *input_tensor = inputs.find(input_name);
      if (input_tensor == nullptr) {
        LOG_ERROR_S << "Input tensor '" << input_name
                    << "' not found in provided inputs for NCNN model: "
                    << m_params.name;
        return InferErrorCode::InferFailed;
      }
      const TypedBuffer &buffer = input_tensor->buffer;
      if (buffer.getSizeBytes() == 0) {
        LOG_ERROR_S << "Empty input data for input tensor '" << input_name;
        return InferErrorCode::InferFailed;
      }
      if (input_tensor->shape.empty()) {
        LOG_ERROR_S << "Shape for input tensor '" << input_name
                    << "' not found or is empty for NCNN model: "
                    << m_params.name;
        return InferErrorCode::InferFailed;
      }
      const std::vector<int> &shape = input_tensor->shape;

      if (buffer.dataType() != DataType::FLOAT32) {
        LOG_ERROR_S << "Unsupported data type for NCNN input '" << input_name
                    << "'. Expected FLOAT32.";
        return InferErrorCode::InferFailed;
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
        LOG_ERROR_S << "Unsupported input shape dimension " << shape.size()
                    << " for NCNN input '" << input_name << "'.";
        return InferErrorCode::InferFailed;
      }

      if (ncnn_in.empty()) {
        LOG_ERROR_S << "Failed to create ncnn::Mat for input: " << input_name;
        return InferErrorCode::InferFailed;
      }
      ex.input(input_name.c_str(), ncnn_in);
    }

    for (const auto &output_name : m_outputNames) {
      ncnn::Mat ncnn_out;
      int ret = ex.extract(output_name.c_str(), ncnn_out);
      if (ret != 0) {
        LOG_ERROR_S << "Failed to extract output '" << output_name
                    << "' from NCNN model: " << m_params.name;
        return InferErrorCode::InferFailed;
      }

      TypedBuffer output_buffer;
      output_buffer.resizeDiscard(ncnn_out.total());
      memcpy(output_buffer.getRawHostPtr(), ncnn_out.data,
             output_buffer.getSizeBytes());

      std::vector<int> shape_vec;
      if (ncnn_out.dims == 1) { // Typically (Features) or (Width)
        shape_vec = {ncnn_out.w};
      } else if (ncnn_out.dims == 2) { // Typically (Height, Width)
        shape_vec = {ncnn_out.h, ncnn_out.w};
      } else if (ncnn_out.dims == 3) { // Typically (Channels, Height, Width)
        shape_vec = {ncnn_out.c, ncnn_out.h, ncnn_out.w};
      } else if (ncnn_out.dims == 4) { // (Channels, Depth, Height, Width)
        shape_vec = {ncnn_out.c, static_cast<int>(ncnn_out.elemsize),
                     ncnn_out.h, ncnn_out.w};
        // representation
        // A common interpretation for ncnn output with dims=4 is (N, C, H, W)
        // where N=1 and d is used for C if cdim is implicit.
        // For safety, let's assume it's (ncnn_out.c, ncnn_out.d, ncnn_out.h,
        // ncnn_out.w)
        // The actual interpretation depends on how the model was constructed.
        // For now, let's use a simpler representation.
        LOG_WARNING_S
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
        shape_vec = {ncnn_out.c, ncnn_out.d, ncnn_out.h, ncnn_out.w};
      }
      outputs.set(output_name, std::move(output_buffer), std::move(shape_vec));
    }
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Std Exception during inference for " << m_params.name
                << ": " << e.what();
    return InferErrorCode::InferFailed;
  } catch (...) {
    LOG_ERROR_S << "Unknown exception during NCNN inference for "
                << m_params.name;
    return InferErrorCode::InferFailed;
  }
}

const ModelInfo &NCNNAlgoInference::getModelInfo() {
  if (m_isInitialized.load(std::memory_order_acquire) && m_modelInfo) {
    std::lock_guard lock(m_mtx);
    if (m_modelInfo) {
      return *m_modelInfo;
    }
  }

  {
    std::lock_guard lock(m_mtx);
    if (!m_isInitialized.load(std::memory_order_relaxed)) {
      LOG_ERROR_S
          << "getModelInfo called on uninitialized or terminated model.";
      static ModelInfo empty_info = {};
      return empty_info;
    }

    if (m_modelInfo) {
      return *m_modelInfo;
    }

    LOG_INFO_S << "Generating model info for: " << m_params.name;
    m_modelInfo = std::make_shared<ModelInfo>();
    m_modelInfo->name = m_params.name;

    m_modelInfo->inputs.reserve(m_inputNames.size());
    for (const auto &input_name : m_inputNames) {
      m_modelInfo->inputs.push_back({input_name, {}});
    }

    m_modelInfo->outputs.reserve(m_outputNames.size());
    for (const auto &output_name : m_outputNames) {
      m_modelInfo->outputs.push_back({output_name, {}});
    }

    return *m_modelInfo;
  }
}

InferErrorCode NCNNAlgoInference::terminate() {
  std::lock_guard<std::mutex> lock(m_mtx);
  LOG_INFO_S << "Terminating NCNN model: " << m_params.name;
  try {
    m_net.clear();                    // Releases network structure and weights
    m_blobPoolAllocator.clear();      // Clear blob memory pool
    m_workspacePoolAllocator.clear(); // Clear workspace memory pool

    for (void *ptr : m_pAlignedBuffers) {
#ifdef _WIN32
      _aligned_free(ptr);
#else
      free(ptr);
#endif
    }
    m_pAlignedBuffers.clear();

    m_inputNames.clear();
    m_outputNames.clear();
    m_modelInfo.reset(); // Release shared_ptr

    LOG_INFO_S << "NCNN Model terminated successfully: " << m_params.name;
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Std Exception during termination for " << m_params.name
                << ": " << e.what();
    return InferErrorCode::TerminateFailed;
  } catch (...) {
    LOG_ERROR_S << "Unknown exception during NCNN model termination: "
                << m_params.name;
    return InferErrorCode::TerminateFailed;
  }
}
}; // namespace ai_core::dnn
