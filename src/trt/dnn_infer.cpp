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
#include "trt_device_buffer.hpp"
#include "trt_utils.hpp"
#include <filesystem>
#include <fstream>
#include <numeric>

namespace ai_core::dnn {

TrtAlgoInference::TrtAlgoInference(const AlgoConstructParams &params)
    : mParams(params.getParam<AlgoInferParams>("params")) {
  LOG_INFOS << "TrtAlgoInference created for model: " << mParams.name;
}

TrtAlgoInference::~TrtAlgoInference() { terminate(); }

void TrtAlgoInference::releaseResources() {
  LOG_INFOS << "Releasing TensorRT resources for model: " << mParams.name;
  mContext.reset();
  mEngine.reset();
  mRuntime.reset();

  if (mStream) {
    cudaError_t err = cudaStreamDestroy(mStream);
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG_WARNINGS << "Failed to destroy CUDA stream: "
                   << cudaGetErrorString(err);
    }
    mStream = nullptr;
  }
  for (auto &buffer : mManagedBuffers) {
    buffer.release();
  }
  mManagedBuffers.clear();
  mTensorAddressMap.clear();
  mTensorSizeMap.clear();
  modelInfo.reset();
  LOG_INFOS << "TensorRT resources released for model: " << mParams.name;
}

InferErrorCode TrtAlgoInference::terminate() {
  std::lock_guard<std::mutex> lock(mMutex);
  if (!mIsInitialized) {
    LOG_INFOS << "TrtAlgoInference terminate called on uninitialized instance: "
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
    LOG_ERRORS << "Model file does not exist: " << path;
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  }

  std::vector<char> engineData;
  if (needsDecrypt) {
    LOG_INFOS << "Decrypting TensorRT engine: " << path;
    std::vector<unsigned char> decryptedData;
    std::string securityKey = SECURITY_KEY;
    auto cryptoConfig = encrypt::Crypto::deriveKeyFromCommit(securityKey);
    encrypt::Crypto crypto(cryptoConfig);
    if (!crypto.decryptData(path, decryptedData)) {
      LOG_ERRORS << "Failed to decrypt model data: " << path;
      return InferErrorCode::INIT_DECRYPTION_FAILED;
    }
    if (decryptedData.empty()) {
      LOG_ERRORS << "Decryption resulted in empty model data: " << path;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    engineData.assign(decryptedData.begin(), decryptedData.end());
  } else {
    std::ifstream engineFile(path, std::ios::binary);
    if (!engineFile) {
      LOG_ERRORS << "Failed to open TensorRT engine file: " << path;
      return InferErrorCode::INIT_MODEL_LOAD_FAILED;
    }
    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    engineData.resize(fileSize);
    engineFile.read(engineData.data(), fileSize);
  }

  if (engineData.empty()) {
    LOG_ERRORS << "Engine data is empty for model: " << path;
    return InferErrorCode::INIT_MODEL_LOAD_FAILED;
  }

  mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
  if (!mRuntime) {
    LOG_ERRORS << "Failed to create TensorRT Runtime.";
    return InferErrorCode::INIT_RUNTIME_FAILED;
  }

  mEngine.reset(
      mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
  if (!mEngine) {
    LOG_ERRORS << "Failed to deserialize TensorRT engine.";
    return InferErrorCode::INIT_ENGINE_FAILED;
  }

  LOG_INFOS << "TensorRT engine loaded and deserialized successfully: "
            << mParams.name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::setupBindings() {
  mContext.reset(mEngine->createExecutionContext());
  if (!mContext) {
    LOG_ERRORS << "Failed to create TensorRT Execution Context.";
    return InferErrorCode::INIT_CONTEXT_FAILED;
  }

  CHECK_CUDA(cudaStreamCreate(&mStream));

  mManagedBuffers.clear();
  mTensorAddressMap.clear();
  mTensorSizeMap.clear();
  modelInfo = std::make_shared<ModelInfo>();
  modelInfo->name = mParams.name;

  const int32_t numIOTensors = mEngine->getNbIOTensors();
  mManagedBuffers.reserve(numIOTensors);

  for (int32_t i = 0; i < numIOTensors; ++i) {
    const char *name = mEngine->getIOTensorName(i);

    auto dims = mEngine->getTensorShape(name);
    auto trtDtype = mEngine->getTensorDataType(name);
    int64_t volume = calculateVolume(dims);
    size_t bufferSize =
        static_cast<size_t>(volume) * trt_utils::getTrtElementSize(trtDtype);

    // Allocate buffer and get pointer
    mManagedBuffers.emplace_back(trt_utils::TrtDeviceBuffer{bufferSize});
    void *devicePtr = mManagedBuffers.back().get();

    // Populate the name-based maps
    mTensorAddressMap[name] = devicePtr;
    mTensorSizeMap[name] = bufferSize;

    // Crucial for enqueueV3
    if (!mContext->setTensorAddress(name, devicePtr)) {
      LOG_ERRORS << "Failed to set tensor address for: " << name;
      return InferErrorCode::INIT_BINDING_FAILED;
    }

    // Populate ModelInfo
    ModelInfo::TensorInfo tensorInfo;
    tensorInfo.name = name;
    tensorInfo.shape.assign(dims.d, dims.d + dims.nbDims);
    tensorInfo.dataType = trt_utils::trtDataTypeToAiCore(trtDtype);

    if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      modelInfo->inputs.push_back(std::move(tensorInfo));
    } else {
      modelInfo->outputs.push_back(std::move(tensorInfo));
    }
  }

  LOG_INFOS
      << "Bindings and buffers configured using name-based API for model: "
      << mParams.name;
  return InferErrorCode::SUCCESS;
}

InferErrorCode TrtAlgoInference::initialize() {
  std::lock_guard<std::mutex> lock(mMutex);
  if (mIsInitialized) {
    LOG_INFOS << "TrtAlgoInference already initialized for model: "
              << mParams.name;
    return InferErrorCode::SUCCESS;
  }

  LOG_INFOS << "Initializing TrtAlgoInference for model: " << mParams.name;

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
    LOG_INFOS << "TrtAlgoInference initialized successfully for model: "
              << mParams.name;
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERRORS << "Exception during initialization: " << e.what();
    releaseResources();
    return InferErrorCode::INIT_FAILED;
  }
}

InferErrorCode TrtAlgoInference::infer(TensorData &inputs,
                                       TensorData &outputs) {
  std::lock_guard<std::mutex> lock(mMutex);
  if (!mIsInitialized) {
    LOG_ERRORS << "Inference called on uninitialized model.";
    return InferErrorCode::NOT_INITIALIZED;
  }

  try {
    // Prepare input buffers: Copy data from user-provided TypedBuffers
    // to the internal device buffers managed by this class.
    for (const auto &inputInfo : modelInfo->inputs) {
      const auto &name = inputInfo.name;
      auto it = inputs.datas.find(name);
      if (it == inputs.datas.end()) {
        LOG_ERRORS << "Input tensor '" << name
                   << "' not found in provided inputs.";
        return InferErrorCode::INFER_INPUT_ERROR;
      }

      TypedBuffer &inputBuffer = it->second;

      const size_t expectedSizeBytes = mTensorSizeMap.at(name);
      const size_t actualSizeBytes = inputBuffer.getSizeBytes();

      if (actualSizeBytes != expectedSizeBytes) {
        LOG_ERRORS << "Mismatched size for input tensor '" << name
                   << "'. Expected: " << expectedSizeBytes
                   << " bytes, Got: " << actualSizeBytes << " bytes.";
        return InferErrorCode::INFER_SIZE_MISMATCH;
      }
      if (inputBuffer.dataType() != inputInfo.dataType) {
        LOG_ERRORS << "Mismatched data type for input tensor '" << name
                   << "'. Expected: " << static_cast<int>(inputInfo.dataType)
                   << ", Got: " << static_cast<int>(inputBuffer.dataType());
        return InferErrorCode::INFER_TYPE_MISMATCH;
      }

      // smart data coyping
      void *destDevicePtr = mTensorAddressMap.at(name);

      if (inputBuffer.location() == BufferLocation::CPU) {
        LOG_INFOS << "Copying CPU input for tensor '" << name << "' (H2D).";
        const void *srcHostPtr = inputBuffer.getRawHostPtr();
        // will adapts automatically to the hardware characteristics of jetson
        CHECK_CUDA(cudaMemcpyAsync(destDevicePtr, srcHostPtr, actualSizeBytes,
                                   cudaMemcpyHostToDevice, mStream));
      } else if (inputBuffer.location() == BufferLocation::GPU_DEVICE) {
        LOG_INFOS << "Copying GPU input for tensor '" << name << "' (D2D).";
        void *srcDevicePtr = inputBuffer.getRawDevicePtr();
        // copy between device and device(pretty fast)
        CHECK_CUDA(cudaMemcpyAsync(destDevicePtr, srcDevicePtr, actualSizeBytes,
                                   cudaMemcpyDeviceToDevice, mStream));

        // For the sake of good portability, the cudaHostAllocMapped mechanism
        // will not be considered for the time being
      } else {
        LOG_ERRORS << "Unsupported buffer location for input tensor: " << name;
        return InferErrorCode::INFER_INVALID_INPUT;
      }
    }

    // Execute Inference
    LOG_INFOS << "Executing inference on stream " << mStream;
    if (!mContext->enqueueV3(mStream)) {
      LOG_ERRORS << "Failed to enqueue TensorRT inference.";
      return InferErrorCode::INFER_EXECUTION_FAILED;
    }

    // Prepare output buffers: Copy data from device to host
    // The output is consistently placed on the CPU for subsequent processing.
    outputs.datas.clear();
    outputs.shapes.clear();
    for (const auto &outputInfo : modelInfo->outputs) {
      const auto &name = outputInfo.name;
      void *srcDevicePtr = mTensorAddressMap.at(name);
      size_t bufferSizeBytes = mTensorSizeMap.at(name);

      // Create a CPU-based TypedBuffer for the output
      std::vector<uint8_t> hostByteVec(bufferSizeBytes);
      void *destinationHostPtr = hostByteVec.data();

      LOG_INFOS << "Copying output for tensor '" << name << "' to CPU (D2H).";
      CHECK_CUDA(cudaMemcpyAsync(destinationHostPtr, srcDevicePtr,
                                 bufferSizeBytes, cudaMemcpyDeviceToHost,
                                 mStream));

      outputs.datas[name] = TypedBuffer::createFromCpu(outputInfo.dataType,
                                                       std::move(hostByteVec));
      outputs.shapes[name].assign(outputInfo.shape.begin(),
                                  outputInfo.shape.end());
    }

    // Synchronize stream to ensure all async operations are complete
    CHECK_CUDA(cudaStreamSynchronize(mStream));

    LOG_INFOS << "Inference and all memory copies synchronized successfully.";
    return InferErrorCode::SUCCESS;
  } catch (const std::exception &e) {
    LOG_ERRORS << "Exception during inference: " << e.what();
    CHECK_CUDA(cudaStreamSynchronize(mStream));
    return InferErrorCode::INFER_FAILED;
  }
}

const ModelInfo &TrtAlgoInference::getModelInfo() {
  if (!mIsInitialized || !modelInfo) {
    LOG_WARNINGS << "getModelInfo() called on uninitialized model.";
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
