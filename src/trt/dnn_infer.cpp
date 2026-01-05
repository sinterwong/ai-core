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

#include "dnn_infer.hpp"
#include "ai_core/infer_error_code.hpp"
#include "crypto.hpp"
#include "cuda_device_buffer.cuh"
#include "trt_utils.hpp"
#include <filesystem>
#include <fstream>
#include <numeric>

namespace ai_core::dnn {

TrtAlgoInference::TrtAlgoInference(const AlgoConstructParams &params)
    : mParams(params.getParam<AlgoInferParams>("params")) {
  LOG_INFO_S << "TrtAlgoInference created for model: " << mParams.name;
}

TrtAlgoInference::~TrtAlgoInference() { terminate(); }

void TrtAlgoInference::releaseResources() {
  LOG_INFO_S << "Releasing TensorRT resources for model: " << mParams.name;
  mContext.reset();
  mEngine.reset();
  mRuntime.reset();

  if (mStream) {
    cudaError_t err = cudaStreamDestroy(mStream);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG_WARNING_S << "Failed to destroy CUDA stream: "
                    << cudaGetErrorString(err);
    }
    mStream = nullptr;
  }
  // CudaDeviceBuffer 会在 clear() 后自动释放内存（析构时）
  mManagedBuffers.clear();
  mTensorAddressMap.clear();
  mTensorSizeMap.clear();
  modelInfo.reset();
  LOG_INFO_S << "TensorRT resources released for model: " << mParams.name;
}

InferErrorCode TrtAlgoInference::terminate() {
  std::lock_guard<std::mutex> lock(mMutex);
  if (!mIsInitialized) {
    LOG_INFO_S
        << "TrtAlgoInference terminate called on uninitialized instance: "
        << mParams.name;
    return InferErrorCode::SUCCESS;
  }
  releaseResources();
  mIsInitialized = false;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::loadEngineFromPath(const std::string &path,
                                                    bool needsDecrypt) {
  if (!std::filesystem::exists(path)) {
    LOG_ERROR_S << "Model file does not exist: " << path;
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  }

  std::vector<char> engineData;
  if (needsDecrypt) {
    LOG_INFO_S << "Decrypting TensorRT engine: " << path;
    std::vector<unsigned char> decryptedData;
    auto cryptoConfig =
        encrypt::Crypto::deriveKeyFromCommit(mParams.decryptkeyStr);
    encrypt::Crypto crypto(cryptoConfig);
    if (!crypto.decryptData(path, decryptedData)) {
      LOG_ERROR_S << "Failed to decrypt model data: " << path;
      return InferErrorCode::INIT_DECRYPTION_FAILED;
    }
    if (decryptedData.empty()) {
      LOG_ERROR_S << "Decryption resulted in empty model data: " << path;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    engineData.assign(decryptedData.begin(), decryptedData.end());
  } else {
    std::ifstream engineFile(path, std::ios::binary);
    if (!engineFile) {
      LOG_ERROR_S << "Failed to open TensorRT engine file: " << path;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    engineData.resize(fileSize);
    engineFile.read(engineData.data(), fileSize);
  }

  if (engineData.empty()) {
    LOG_ERROR_S << "Engine data is empty for model: " << path;
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  }

  mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
  if (!mRuntime) {
    LOG_ERROR_S << "Failed to create TensorRT Runtime.";
    return InferErrorCode::INIT_RUNTIME_FAILED;
  }

  mEngine.reset(
      mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
  if (!mEngine) {
    LOG_ERROR_S << "Failed to deserialize TensorRT engine.";
    return InferErrorCode::INIT_ENGINE_FAILED;
  }

  LOG_INFO_S << "TensorRT engine loaded and deserialized successfully: "
             << mParams.name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::setupBindings() {
  mContext.reset(mEngine->createExecutionContext());
  if (!mContext) {
    LOG_ERROR_S << "Failed to create TensorRT Execution Context.";
    return InferErrorCode::INIT_CONTEXT_FAILED;
  }

  CHECK_CUDA(cudaStreamCreate(&mStream));

  mManagedBuffers.clear();
  mTensorAddressMap.clear();
  mTensorSizeMap.clear();
  modelInfo = std::make_shared<ModelInfo>();
  modelInfo->name = mParams.name;

  const int profileIndex = 0;
  if (mEngine->getNbOptimizationProfiles() <= profileIndex) {
    LOG_ERROR_S << "Engine does not have optimization profile at index "
                << profileIndex;
    return InferErrorCode::INIT_FAILED;
  }
  LOG_INFO_S << "Using optimization profile 0.";

  const int32_t numIOTensors = mEngine->getNbIOTensors();

  // Set Input Shapes to MAX to allow automatic output size deduction
  for (int32_t i = 0; i < numIOTensors; ++i) {
    const char *name = mEngine->getIOTensorName(i);
    if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      // 获取该 Profile 下定义的最大输入尺寸
      auto maxDims = mEngine->getProfileShape(
          name, profileIndex, nvinfer1::OptProfileSelector::kMAX);
      // 将 Context 的输入维度设置为最大，以便后续推导最大输出维度
      if (!mContext->setInputShape(name, maxDims)) {
        LOG_WARNING_S
            << "Failed to set max input shape for auto-sizing tensor: " << name;
      }
    }
  }

  // Allocate Buffers
  mManagedBuffers.reserve(numIOTensors);

  for (int32_t i = 0; i < numIOTensors; ++i) {
    const char *name = mEngine->getIOTensorName(i);
    auto trtDtype = mEngine->getTensorDataType(name);

    // 默认尝试从 Profile 获取 Max Shape
    auto dims = mEngine->getProfileShape(name, profileIndex,
                                         nvinfer1::OptProfileSelector::kMAX);

    int64_t volume = -1;
    size_t bufferSize = 0;

    if (dims.nbDims >= 0) {
      volume = calculateVolume(dims);
    }

    if (volume < 0) {
      // 处理动态维度
      if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        // 优先检查用户是否提供了最大 Buffer 配置
        auto it = mParams.maxOutputBufferSizes.find(name);
        if (it != mParams.maxOutputBufferSizes.end()) {
          bufferSize = it->second;
          LOG_INFO_S << "Output tensor '" << name
                     << "' is dynamic. Using user-configured max buffer size: "
                     << bufferSize << " bytes.";
        }
        // 用户未配置，尝试根据 Context (已设为 Max Input) 推导
        else {
          LOG_INFO_S << "Output tensor '" << name
                     << "' is dynamic and no user config found. "
                     << "Attempting to deduce max shape from context...";

          // 获取由 Max Input 推导出的当前 Output Shape
          nvinfer1::Dims deducedDims = mContext->getTensorShape(name);
          int64_t deducedVolume = calculateVolume(deducedDims);

          if (deducedVolume > 0) {
            bufferSize = static_cast<size_t>(deducedVolume) *
                         trt_utils::getTrtElementSize(trtDtype);
            LOG_INFO_S << "Auto-deduced max buffer size for '" << name
                       << "': " << bufferSize << " bytes (" << deducedVolume
                       << " elements).";
          } else {
            // 或许会因为输出取决于数据的值，而非仅仅取决于输入的维度，依然无法推导（例如NonZero）
            LOG_ERROR_S
                << "Could not deduce max size for dynamic output tensor '"
                << name
                << "'. The shape might be data-dependent (value-dependent). "
                << "Please provide it in "
                   "AlgoInferParams::maxOutputBufferSizes.";
            return InferErrorCode::INIT_BINDING_FAILED;
          }
        }
      } else {
        // Input tensor 必须有有效的 Profile Max Shape
        LOG_ERROR_S << "Input tensor '" << name
                    << "' has an unexpected dynamic dimension (-1) in Profile.";
        return InferErrorCode::INIT_BINDING_FAILED;
      }
    } else {
      // 静态维度或 Profile 中明确定义的 Max 维度
      bufferSize =
          static_cast<size_t>(volume) * trt_utils::getTrtElementSize(trtDtype);
    }

    if (bufferSize == 0) {
      LOG_WARNING_S << "Tensor '" << name << "' has a buffer size of 0.";
    }

    // Allocate buffer using CudaDeviceBuffer and get pointer
    // DeviceByteBuffer 的 size 就是字节数
    mManagedBuffers.emplace_back(cuda_utils::DeviceByteBuffer{bufferSize});
    void *devicePtr = mManagedBuffers.back().unsafePtr();

    // Populate the name-based maps
    mTensorAddressMap[name] = devicePtr;
    mTensorSizeMap[name] = bufferSize;

    // Crucial for enqueueV3
    if (!mContext->setTensorAddress(name, devicePtr)) {
      LOG_ERROR_S << "Failed to set tensor address for: " << name;
      return InferErrorCode::INIT_BINDING_FAILED;
    }

    // Populate ModelInfo
    ModelInfo::TensorInfo tensorInfo;
    tensorInfo.name = name;

    // 这里获取的是 Binding 的原始 Shape (可能包含 -1)
    // 用于告知上层应用哪些维度是动态的
    auto bindingDims = mEngine->getTensorShape(name);
    tensorInfo.shape.assign(bindingDims.d, bindingDims.d + bindingDims.nbDims);
    tensorInfo.dataType = trt_utils::trtDataTypeToAiCore(trtDtype);

    if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      for (const auto &dim : tensorInfo.shape) {
        if (dim == -1) {
          mDynamicInputTensorNames.insert(name);
          LOG_DEBUG_S << "Input tensor '" << name
                      << "' is identified as dynamic.";
          break;
        }
      }
      modelInfo->inputs.emplace_back(std::move(tensorInfo));
    } else {
      modelInfo->outputs.emplace_back(std::move(tensorInfo));
    }
  }

  LOG_INFO_S
      << "Bindings and buffers configured using name-based API for model: "
      << mParams.name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::initialize() {
  std::lock_guard<std::mutex> lock(mMutex);
  if (mIsInitialized) {
    LOG_INFO_S << "TrtAlgoInference already initialized for model: "
               << mParams.name;
    return InferErrorCode::SUCCESS;
  }

  LOG_INFO_S << "Initializing TrtAlgoInference for model: " << mParams.name;

  try {
    InferErrorCode err =
        loadEngineFromPath(mParams.modelPath, mParams.needDecrypt);
    if (err != InferErrorCode::SUCCESS) {
      releaseResources();
      return err;
    }

    err = setupBindings();
    if (err != InferErrorCode::SUCCESS) {
      releaseResources();
      return err;
    }

    mIsInitialized = true;
    LOG_INFO_S << "TrtAlgoInference initialized successfully for model: "
               << mParams.name;
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception during initialization: " << e.what();
    releaseResources();
    return InferErrorCode::INIT_FAILED;
  }
}

InferErrorCode TrtAlgoInference::infer(const TensorData &inputs,
                                       TensorData &outputs) {
  std::lock_guard<std::mutex> lock(mMutex);
  if (!mIsInitialized) {
    LOG_ERROR_S << "Inference called on uninitialized model.";
    return InferErrorCode::NOT_INITIALIZED;
  }

  try {
    // Prepare input buffers: Copy data from user-provided TypedBuffers
    // to the internal device buffers managed by this class.
    for (const auto &inputInfo : modelInfo->inputs) {
      const auto &name = inputInfo.name;
      auto dataIt = inputs.datas.find(name);
      if (dataIt == inputs.datas.end()) {
        LOG_ERROR_S << "Input tensor '" << name
                    << "' not found in provided inputs.";
        return InferErrorCode::INFER_INPUT_ERROR;
      }

      const TypedBuffer &inputBuffer = dataIt->second;
      if (inputBuffer.dataType() != inputInfo.dataType) {
        LOG_ERROR_S << "Mismatched data type for input tensor '" << name
                    << "'. Expected: " << static_cast<int>(inputInfo.dataType)
                    << ", Got: " << static_cast<int>(inputBuffer.dataType());
        return InferErrorCode::INFER_TYPE_MISMATCH;
      }

      const size_t actualSizeBytes = inputBuffer.getSizeBytes();
      const bool isDynamic = mDynamicInputTensorNames.count(name);
      if (isDynamic) {
        auto shapeIt = inputs.shapes.find(name);
        if (shapeIt == inputs.shapes.end()) {
          LOG_ERROR_S << "Shape info for dynamic input tensor '" << name
                      << "' must be provided.";
          return InferErrorCode::INFER_INPUT_ERROR;
        }

        const auto &actualShapeVec = shapeIt->second;
        nvinfer1::Dims actualDims;
        actualDims.nbDims = actualShapeVec.size();
        std::copy(actualShapeVec.begin(), actualShapeVec.end(), actualDims.d);

        if (!mContext->setInputShape(name.c_str(), actualDims)) {
          LOG_ERROR_S << "Failed to set input shape for tensor: " << name;
          return InferErrorCode::INFER_EXECUTION_FAILED;
        }
        if (actualSizeBytes > mTensorSizeMap.at(name)) {
          LOG_ERROR_S << "Actual size for dynamic input '" << name
                      << "' exceeds max buffer size.";
          return InferErrorCode::INFER_SIZE_MISMATCH;
        }
      } else {
        if (actualSizeBytes != mTensorSizeMap.at(name)) {
          LOG_ERROR_S << "Mismatched size for static input tensor '" << name
                      << "'. Expected: " << mTensorSizeMap.at(name)
                      << " bytes, Got: " << actualSizeBytes << " bytes.";
          return InferErrorCode::INFER_SIZE_MISMATCH;
        }
      }

      // smart data coyping
      void *destDevicePtr = mTensorAddressMap.at(name);
      if (inputBuffer.location() == BufferLocation::CPU) {
        LOG_TRACE_S << "Copying CPU input for tensor '" << name << "' (H2D).";
        const void *srcHostPtr = inputBuffer.getRawHostPtr();
        // will adapts automatically to the hardware characteristics of jetson
        CHECK_CUDA(cudaMemcpyAsync(destDevicePtr, srcHostPtr, actualSizeBytes,
                                   cudaMemcpyHostToDevice, mStream));
      } else if (inputBuffer.location() == BufferLocation::GPU_DEVICE) {
        LOG_TRACE_S << "Copying GPU input for tensor '" << name << "' (D2D).";
        void *srcDevicePtr = inputBuffer.getRawDevicePtr();
        // copy between device and device(pretty fast)
        CHECK_CUDA(cudaMemcpyAsync(destDevicePtr, srcDevicePtr, actualSizeBytes,
                                   cudaMemcpyDeviceToDevice, mStream));
        // For the sake of good portability, the cudaHostAllocMapped mechanism
        // will not be considered for the time being
      } else {
        LOG_ERROR_S << "Unsupported buffer location for input tensor: " << name;
        return InferErrorCode::INFER_INVALID_INPUT;
      }
    }

    // Execute Inference
    LOG_TRACE_S << "Executing inference on stream " << mStream;
    if (!mContext->enqueueV3(mStream)) {
      LOG_ERROR_S << "Failed to enqueue TensorRT inference.";
      return InferErrorCode::INFER_EXECUTION_FAILED;
    }

    // Prepare output buffers: Copy data from device to host
    // The output is consistently placed on the CPU for subsequent processing.
    outputs.datas.clear();
    outputs.shapes.clear();
    for (const auto &outputInfo : modelInfo->outputs) {
      const auto &name = outputInfo.name;
      void *srcDevicePtr = mTensorAddressMap.at(name);

      nvinfer1::Dims actualOutputDims = mContext->getTensorShape(name.c_str());
      int64_t actualVolume = calculateVolume(actualOutputDims);

      if (actualVolume < 0) {
        LOG_ERROR_S
            << "Inference resulted in invalid output dimensions for tensor: "
            << name;
        return InferErrorCode::INFER_EXECUTION_FAILED;
      }

      size_t actualOutputSizeBytes =
          static_cast<size_t>(actualVolume) *
          trt_utils::getTrtElementSize(
              trt_utils::aiCoreDataTypeToTrt(outputInfo.dataType));

      // Create a CPU-based TypedBuffer for the output
      std::vector<uint8_t> hostByteVec(actualOutputSizeBytes);
      void *destinationHostPtr = hostByteVec.data();

      LOG_TRACE_S << "Copying output for tensor '" << name << "' to CPU (D2H).";
      CHECK_CUDA(cudaMemcpyAsync(destinationHostPtr, srcDevicePtr,
                                 actualOutputSizeBytes, cudaMemcpyDeviceToHost,
                                 mStream));

      outputs.datas[name] = TypedBuffer::createFromCpu(outputInfo.dataType,
                                                       std::move(hostByteVec));
      outputs.shapes[name].assign(actualOutputDims.d,
                                  actualOutputDims.d + actualOutputDims.nbDims);
    }

    // Synchronize stream to ensure all async operations are complete
    CHECK_CUDA(cudaStreamSynchronize(mStream));

    LOG_TRACE_S << "Inference and all memory copies synchronized successfully.";
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERROR_S << "Exception during inference: " << e.what();
    CHECK_CUDA(cudaStreamSynchronize(mStream));
    return InferErrorCode::INFER_FAILED;
  }
}

const ModelInfo &TrtAlgoInference::getModelInfo() {
  if (!mIsInitialized || !modelInfo) {
    LOG_WARNING_S << "getModelInfo() called on uninitialized model.";
    static ModelInfo emptyInfo;
    return emptyInfo;
  }
  return *modelInfo;
}

int64_t TrtAlgoInference::calculateVolume(const nvinfer1::Dims &dims) {
  return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1},
                         std::multiplies<int64_t>());
}

}; // namespace ai_core::dnn