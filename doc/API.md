# AI Core Framework API Reference

## 1. Overview

The AI Core Framework is a modular, extensible, and high-performance C++ framework designed for building and deploying AI inference pipelines. Its core architecture is based on the principle of decoupling the three main stages of an AI task:

1.  **Pre-processing**: Transforming raw input (like images) into a model-ready format.
2.  **Inference**: Executing the neural network model on a specific hardware backend (CPU/GPU).
3.  **Post-processing**: Parsing the raw model output into structured, human-readable results.

This design allows for maximum code reuse and flexibility, enabling developers to easily assemble new pipelines or swap out components (e.g., changing the inference backend from ONNX Runtime to TensorRT) with minimal code changes.

## 2. Core Concepts

### The Three-Stage Pipeline

A typical inference flow follows a clear, linear path, orchestrated by the `AlgoInference` class.

`AlgoInput` -> **[Pre-process]** -> `TensorData` -> **[Inference Engine]** -> `TensorData` -> **[Post-process]** -> `AlgoOutput`

### Type-Safe Data Handling

The framework avoids `void*` and unsafe casting by leveraging `std::variant` and a wrapper class `ParamCenter`. This ensures that all data types for inputs, outputs, and parameters are handled in a type-safe and modern C++ manner.

## 3. High-Level API

These are the primary classes for managing and executing algorithm pipelines.

### `ai_core::dnn::AlgoInference`

This class represents a single, complete end-to-end algorithm pipeline. It encapsulates one pre-processing module, one inference engine, and one post-processing module.

**Constructor:**

*   `AlgoInference(const AlgoModuleTypes &moduleTypes, const AlgoInferParams &inferParams)`
    *   Constructs a pipeline by specifying the names of the modules to use (see Plugin System) and the inference parameters.

**Key Methods:**

*   `InferErrorCode initialize()`
    *   Initializes all internal modules (pre-processing, inference engine, post-processing). Must be called before `infer`.
*   `InferErrorCode infer(const AlgoInput &input, const AlgoPreprocParams &preprocParams, const AlgoPostprocParams &postprocParams, AlgoOutput &output)`
    *   Executes the single-item inference pipeline.
*   `InferErrorCode batchInfer(...)`
    *   Executes the batch inference pipeline.
*   `InferErrorCode terminate()`
    *   Releases all resources.

## 4. Key Data Structures

These are the data structures used to pass information between the user and the framework.

### 4.1. General Purpose Data Wrappers

These wrappers use `ParamCenter<std::variant<...>>` to hold different types of data.

*   `ai_core::AlgoInput`: The input to a pipeline. Can contain:
    *   `FrameInput`: For single image-based tasks.
    *   `FrameInputWithMask`: For tasks requiring an image and associated mask regions.
*   `ai_core::AlgoOutput`: The final result from a pipeline. Can contain:
    *   `ClsRet`: Classification result.
    *   `DetRet`: Object detection result.
    *   `SegRet`: Semantic segmentation result.
    *   And others like `FeatureRet`, `FprClsRet`, etc.
*   `ai_core::AlgoPreprocParams`: Configuration for the pre-processing stage. Can contain:
    *   `FramePreprocessArg`: Common parameters for image pre-processing (resize, normalization, etc.).
*   `ai_core::AlgoPostprocParams`: Configuration for the post-processing stage. Can contain:
    *   `AnchorDetParams`: Parameters for anchor-based detection models.
    *   `GenericPostParams`: Generic parameters.
    *   `ConfidenceFilterParams`: Simple confidence thresholding parameters.

### 4.2. Core Data Containers

*   `ai_core::TensorData`
    *   The standardized data structure for model inputs and outputs. It contains:
        *   `std::map<std::string, TypedBuffer> datas`: Maps tensor names to their data buffers.
        *   `std::map<std::string, std::vector<int>> shapes`: Maps tensor names to their shapes.
*   `ai_core::TypedBuffer`
    *   A powerful abstraction for memory buffers that can reside on either the CPU or a GPU device. It handles memory management and provides safe access to the underlying data.
    *   **Key Properties:**
        *   `DataType dataType()`: The data type of elements in the buffer (e.g., `FLOAT32`, `INT8`).
        *   `BufferLocation location()`: The location of the buffer (`CPU` or `GPU_DEVICE`).
    *   **Key Methods:**
        *   `static TypedBuffer createFromCpu(...)`: Creates a buffer from a `std::vector<uint8_t>` on the CPU.
        *   `static TypedBuffer createFromGpu(...)`: Creates a buffer on the specified GPU device.
        *   `template <typename T> const T* getHostPtr() const`: Returns a typed pointer to the CPU data. Throws if the buffer is on the GPU.
        *   `void* getRawDevicePtr() const`: Returns a raw `void*` to the GPU memory.

### 4.3. Parameter Structures

*   `ai_core::AlgoInferParams`: Main configuration for the inference engine.
    *   `name`: The name of the algorithm instance.
    *   `modelPath`: Path to the model file.
    *   `deviceType`: `DeviceType::CPU` or `DeviceType::GPU`.
    *   `dataType`: The primary data type for inference (e.g., `DataType::FLOAT32`).
    *   `maxOutputBufferSizes`: Pre-allocates buffer sizes for output tensors.

### 4.4. Common Types

Defined in `ai_core/infer_common_types.hpp`.

*   `enum class DeviceType`: `CPU`, `GPU`.
*   `enum class DataType`: `FLOAT32`, `FLOAT16`, `INT32`, `INT8`, etc.
*   `struct ModelInfo`: Contains metadata about the model, such as input/output tensor names, shapes, and data types.

## 5. Extensibility: The Plugin System

The framework uses a **Self-Registering Factory** pattern to make components pluggable. You can add new pre-processing, inference, or post-processing modules without modifying the framework's core code.

There are three main factories:
*   `ai_core::dnn::PreprocFactory`
*   `ai_core::dnn::InferEngineFactory`
*   `ai_core::dnn::PostprocFactory`

To add a new component (e.g., a new post-processing algorithm named `MyPostproc`):

1.  Inherit from the corresponding base class (e.g., `PostprocssBase`).
2.  Implement the required virtual methods.
3.  In a `.cpp` file, register your new class using the provided macro:

    ```cpp
    // In my_postproc.cpp
    #include "postproc_registrar.hpp"
    
    // ... MyPostproc class implementation ...
    
    // This line automatically registers the class with the factory
    REGISTER_POSTPROCESS_ALGO(MyPostproc); 
    ```

Now, you can specify `"MyPostproc"` in the `AlgoModuleTypes` structure to use it in a pipeline.

## 6. Error Handling

Most functions in the framework return an `ai_core::InferErrorCode`.

*   `InferErrorCode::SUCCESS` (value 0) indicates success.
*   Any other value indicates a failure. The error codes are grouped by category:
    *   `INIT_*`: Errors during initialization.
    *   `INFER_*`: Errors during the inference process.
    *   `ALGO_*`: Errors related to the `AlgoManager`.

Always check the return code of the framework's functions.
