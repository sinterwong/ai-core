# AI Core API 参考手册

欢迎使用 AI Core 框架的 API 参考手册。本文档详细介绍了用于构建 AI 推理应用的所有核心组件、数据类型和函数。

## 目录

- [1. 核心推理流程 - `AlgoInference`](#1-核心推理流程---algoinference)
  - [1.1. 概述](#11-概述)
  - [1.2. 构造函数](#12-构造函数)
  - [1.3. 核心方法](#13-核心方法)
  - [1.4. 完整使用示例](#14-完整使用示例)
- [2. 核心数据类型](#2-核心数据类型)
  - [2.1. `AlgoInput` - 算法输入](#21-algoinput---算法输入)
  - [2.2. `AlgoOutput` - 算法输出](#22-algooutput---算法输出)
  - [2.3. `AlgoPreprocParams` - 预处理参数](#23-algopreprocparams---预处理参数)
  - [2.4. `AlgoPostprocParams` - 后处理参数](#24-algopostprocparams---后处理参数)
- [3. 核心数据容器](#3-核心数据容器)
  - [3.1. `TypedBuffer` - 类型化内存缓冲区](#31-typedbuffer---类型化内存缓冲区)
  - [3.2. `TensorData` - 张量数据集合](#32-tensordata---张量数据集合)
- [4. 通用数据结构](#4-通用数据结构)
  - [4.1. 输入数据结构](#41-输入数据结构)
  - [4.2. 输出数据结构](#42-输出数据结构)
  - [4.3. 参数数据结构](#43-参数数据结构)
- [5. 枚举类型](#5-枚举类型)
  - [5.1. `InferErrorCode` - 错误码](#51-infererrorcode---错误码)
  - [5.2. `DeviceType` - 设备类型](#52-devicetype---设备类型)
  - [5.3. `DataType` - 数据类型](#53-datatype---数据类型)
  - [5.4. `BufferLocation` - 缓冲区位置](#54-bufferlocation---缓冲区位置)
- [6. 框架扩展 (插件开发)](#6-框架扩展-插件开发)
  - [6.1. 插件接口](#61-插件接口)
  - [6.2. 插件注册](#62-插件注册)

---

## 1. 核心推理流程 - `AlgoInference`

这是框架最主要的入口类，封装了完整的 **预处理 -> 推理 -> 后处理** 流水线。

**头文件:** `#include "ai_core/algo_infer.hpp"`

### 1.1. 概述

`AlgoInference` 负责管理整个推理生命周期，包括加载插件、初始化引擎、执行推理和释放资源。

### 1.2. 构造函数

```cpp
AlgoInference(const AlgoModuleTypes &moduleTypes,
              const AlgoInferParams &inferParams);
```

- `moduleTypes`: 一个 `AlgoModuleTypes` 结构体，用于指定流水线中每个阶段所使用的插件名称（字符串）。
  ```cpp
  struct AlgoModuleTypes {
    std::string preprocModule;
    std::string inferModule;
    std::string postprocModule;
  };
  ```
- `inferParams`: 一个 `AlgoInferParams` 结构体，用于配置推理引擎，如模型路径、设备类型等。

### 1.3. 核心方法

- **`InferErrorCode initialize()`**
  初始化整个推理流水线，包括加载和初始化所有指定的插件。必须在调用 `infer` 之前成功调用。
  - **返回:** `InferErrorCode::SUCCESS` 表示成功。

- **`InferErrorCode infer(const AlgoInput &input, const AlgoPreprocParams &preprocParams, const AlgoPostprocParams &postprocParams, AlgoOutput &output)`**
  执行单次推理。
  - `input`: 算法输入数据 (`AlgoInput`)。
  - `preprocParams`: 预处理参数 (`AlgoPreprocParams`)。
  - `postprocParams`: 后处理参数 (`AlgoPostprocParams`)。
  - `output`: 用于接收算法输出结果 (`AlgoOutput`)。
  - **返回:** `InferErrorCode::SUCCESS` 表示成功。

- **`InferErrorCode batchInfer(const std::vector<AlgoInput> &inputs, const AlgoPreprocParams &preprocParams, const AlgoPostprocParams &postprocParams, std::vector<AlgoOutput> &outputs)`**
  执行批量推理。
  - `inputs`: 一组算法输入数据。
  - `outputs`: 用于接收一组算法输出结果。
  - **返回:** `InferErrorCode::SUCCESS` 表示成功。

- **`InferErrorCode terminate()`**
  释放所有资源，终止推理流水线。
  - **返回:** `InferErrorCode::SUCCESS` 表示成功。

- **`const ModelInfo &getModelInfo() const noexcept`**
  获取当前加载模型的信息，包括输入输出张量的名称、形状和数据类型。

### 1.4. 完整使用示例

```cpp
#include "ai_core/algo_infer.hpp"

// ...

// 1. 定义模块和算法参数
ai_core::AlgoModuleTypes modules = {"MyPreproc", "MyInferEngine", "MyPostproc"};
ai_core::AlgoInferParams params = {/* ... */};

// 2. 创建和初始化
ai_core::dnn::AlgoInference pipeline(modules, params);
if (pipeline.initialize() != ai_core::InferErrorCode::SUCCESS) {
    // 错误处理
}

// 3. 准备数据
ai_core::AlgoInput input;
input.setParams(ai_core::FrameInput{...});

ai_core::AlgoPreprocParams preproc_args;
preproc_args.setParams(ai_core::FramePreprocessArg{...});

ai_core::AlgoPostprocParams postproc_args;
postproc_args.setParams(ai_core::AnchorDetParams{...});

// 4. 推理
ai_core::AlgoOutput output;
pipeline.infer(input, preproc_args, postproc_args, output);

// 5. 解析结果
if (auto* result = output.getParams<ai_core::DetRet>()) {...}

// 6. 释放资源
pipeline.terminate();
```

---

## 2. 核心数据类型

这些类型是 `AlgoInference::infer` 方法的参数，它们使用 `ParamCenter` 模板和 `std::variant` 来实现类型安全和灵活性。

**核心思想:** 使用 `setParams<T>(...)` 存入具体类型的参数，使用 `getParams<T>()` 取出。

### 2.1. `AlgoInput` - 算法输入

**定义:** `using AlgoInput = ParamCenter<std::variant<std::monostate, FrameInput, FrameInputWithMask>>;`

用于包装实际的输入数据。

- **可包含类型:**
  - `FrameInput`: 单帧图像输入。
  - `FrameInputWithMask`: 带掩码区域的单帧图像输入。

**示例:**
```cpp
#include "ai_core/algo_data_types.hpp"

ai_core::AlgoInput input;
auto image = std::make_shared<cv::Mat>(cv::imread("test.jpg"));
input.setParams(ai_core::FrameInput{image, nullptr});
```

### 2.2. `AlgoOutput` - 算法输出

**定义:** `using AlgoOutput = ParamCenter<std::variant<std::monostate, ClsRet, DetRet, ...>>;`

用于接收算法的输出结果。

- **可包含类型:**
  - `ClsRet`: 分类结果。
  - `DetRet`: 检测结果。
  - `FeatureRet`: 特征向量结果。
  - `SegRet`: 分割结果。
  - ... (更多请参见 [4.2. 输出数据结构](#42-输出数据结构))

**示例:**
```cpp
ai_core::AlgoOutput output;
// ... 调用 infer 填充 output ...

// 尝试获取 DetRet 类型的结果
if (auto* det_result = output.getParams<ai_core::DetRet>()) {
    std::cout << "Found " << det_result->bboxes.size() << " objects." << std::endl;
} else if (auto* cls_result = output.getParams<ai_core::ClsRet>()) {
    std::cout << "Class: " << cls_result->label << ", Score: " << cls_result->score << std::endl;
}
```

### 2.3. `AlgoPreprocParams` - 预处理参数

**定义:** `using AlgoPreprocParams = ParamCenter<std::variant<std::monostate, FramePreprocessArg>>;`

- **可包含类型:**
  - `FramePreprocessArg`: 通用的图像预处理参数。

### 2.4. `AlgoPostprocParams` - 后处理参数

**定义:** `using AlgoPostprocParams = ParamCenter<std::variant<std::monostate, AnchorDetParams, GenericPostParams, ConfidenceFilterParams>>;`

- **可包含类型:**
  - `AnchorDetParams`: 用于基于 Anchor 的目标检测算法的后处理参数。
  - `GenericPostParams`: 通用后处理参数，如 Softmax 分类。
  - `ConfidenceFilterParams`: 用于分割等任务的置信度过滤参数。

---

## 3. 核心数据容器

### 3.1. `TypedBuffer` - 类型化内存缓冲区

**头文件:** `#include "ai_core/typed_buffer.hpp"`

`TypedBuffer` 是一个强大的数据容器，它抽象了 CPU 和 GPU 内存，并关联了数据类型。

- **创建 (静态工厂方法):**
  - `static TypedBuffer createFromCpu(DataType type, const std::vector<uint8_t>& data)`: 从已有的 CPU 数据创建（拷贝）。
  - `static TypedBuffer createFromCpu(DataType type, std::vector<uint8_t>&& data)`: 从已有的 CPU 数据创建（移动）。
  - `static TypedBuffer createFromGpu(DataType type, size_t sizeBytes, int deviceId = 0)`: 在 GPU 上分配指定大小的内存。
  - `static TypedBuffer createFromGpu(DataType type, void* devicePtr, ..., bool manageMemory)`: 从一个外部 GPU 指针创建，可选择是否让 `TypedBuffer` 管理该内存的生命周期。

- **数据访问:**
  - `template <typename T> const T* getHostPtr() const`: 获取 CPU 数据的只读指针。如果数据在 GPU 上或类型不匹配，会抛出异常。
  - `template <typename T> T* getHostPtr()`: 获取 CPU 数据的可写指针。
  - `void* getRawDevicePtr() const`: 获取 GPU 数据的 `void*` 指针。

- **属性查询:**
  - `DataType dataType() const`: 获取数据类型。
  - `BufferLocation location() const`: 获取数据位置 (CPU/GPU)。
  - `size_t getSizeBytes() const`: 获取缓冲区总字节数。
  - `size_t getElementCount() const`: 获取缓冲区内的元素数量。
  - `int getDeviceId() const`: 获取 GPU 设备 ID。

**示例:**
```cpp
// 从 vector 创建一个 float32 类型的 CPU buffer
std::vector<float> my_floats = {1.0f, 2.0f, 3.0f};
std::vector<uint8_t> my_bytes(
    reinterpret_cast<uint8_t*>(my_floats.data()),
    reinterpret_cast<uint8_t*>(my_floats.data()) + my_floats.size() * sizeof(float)
);
auto cpu_buffer = ai_core::TypedBuffer::createFromCpu(ai_core::DataType::FLOAT32, std::move(my_bytes));

// 访问数据
const float* data_ptr = cpu_buffer.getHostPtr<float>();
std::cout << "First element: " << data_ptr[0] << std::endl;

// 创建一个 1MB 大小的 GPU buffer
auto gpu_buffer = ai_core::TypedBuffer::createFromGpu(ai_core::DataType::INT8, 1024 * 1024, 0);
void* device_ptr = gpu_buffer.getRawDevicePtr();
```

### 3.2. `TensorData` - 张量数据集合

**头文件:** `#include "ai_core/tensor_data.hpp"`

`TensorData` 是流水线内部用于传递数据的标准格式，它代表了一组命名的张量。

```cpp
struct TensorData {
  // 张量名到数据缓冲区的映射
  std::map<std::string, TypedBuffer> datas;
  // 张量名到形状的映射
  std::map<std::string, std::vector<int>> shapes;
};
```
> **注意:** 应用开发者通常不需要直接创建 `TensorData`，它主要由预处理、推理和后处理插件在内部使用和传递。

---

## 4. 通用数据结构

### 4.1. 输入数据结构

- **`FrameInput`** (`algo_input_types.hpp`)
  ```cpp
  struct FrameInput {
    std::shared_ptr<cv::Mat> image;      // 原始图像
    std::shared_ptr<cv::Rect> inputRoi; // 感兴趣区域 (可选)
  };
  ```

### 4.2. 输出数据结构

- **`BBox`** (`algo_output_types.hpp`)
  ```cpp
  struct BBox {
    std::shared_ptr<cv::Rect> rect; // 边界框
    float score;                    // 置信度
    int label;                      // 类别标签
  };
  ```
- **`DetRet`** (`algo_output_types.hpp`)
  ```cpp
  struct DetRet {
    std::vector<BBox> bboxes; // 检测到的所有边界框
  };
  ```
- **`ClsRet`** (`algo_output_types.hpp`)
  ```cpp
  struct ClsRet {
    float score; // 置信度
    int label;   // 类别标签
  };
  ```
- **`FeatureRet`** (`algo_output_types.hpp`)
  ```cpp
  struct FeatureRet {
    std::vector<float> feature; // 特征向量
    int featSize;               // 特征维度
  };
  ```
- ... 更多请参考 `ai_core/algo_output_types.hpp`。

### 4.3. 参数数据结构

- **`FramePreprocessArg`** (`preproc_types.hpp`)
  定义了图像预处理的所有参数，如目标尺寸、均值/归一化系数、颜色通道顺序等。
- **`AnchorDetParams`** (`postproc_types.hpp`)
  定义了 Anchor-based 检测算法的后处理参数，如置信度阈值、NMS 阈值等。
- **`GenericPostParams`** (`postproc_types.hpp`)
  定义了通用后处理算法（如 Softmax）的参数。
- ... 更多请参考 `ai_core/preproc_types.hpp` 和 `ai_core/postproc_types.hpp`。

---

## 5. 枚举类型

### 5.1. `InferErrorCode` - 错误码

**头文件:** `ai_core/infer_error_code.hpp`
定义了框架中所有可能返回的错误码。

- `SUCCESS = 0`: 操作成功。
- `INIT_FAILED = 100`: 初始化失败。
- `INIT_MODEL_LOAD_FAILED = 102`: 模型加载失败。
- `INFER_FAILED = 200`: 推理执行失败。
- `INFER_INPUT_ERROR = 201`: 推理输入数据错误。
- `ALGO_NOT_FOUND = 400`: 找不到指定的算法插件。
- ... 更多请参考头文件。

### 5.2. `DeviceType` - 设备类型

**头文件:** `ai_core/infer_common_types.hpp`
- `CPU = 0`
- `GPU = 1`

### 5.3. `DataType` - 数据类型

**头文件:** `ai_core/infer_common_types.hpp`
- `FLOAT32 = 0`
- `FLOAT16`
- `INT32`
- `INT64`
- `INT8`

### 5.4. `BufferLocation` - 缓冲区位置

**头文件:** `ai_core/typed_buffer.hpp`
- `CPU`
- `GPU_DEVICE`

---

## 6. 自定义插件开发

本节面向希望自定义插件的开发者。

### 6.1. 插件接口

要扩展框架，你需要实现以下接口之一：
- `ai_core::dnn::IPreprocssPlugin`: 预处理插件接口。
- `ai_core::dnn::IInferEnginePlugin`: 推理引擎插件接口。
- `ai_core::dnn::IPostprocssPlugin`: 后处理插件接口。

你需要继承相应的基类并实现其纯虚函数，例如 `process()` 或 `infer()`。

### 6.2. 插件注册

实现插件后，需要使用注册宏将其注册到框架的工厂中，以便 `AlgoInference` 可以通过名称找到它。

**头文件:** `ai_core/ai_core_registrar.hpp`

- **`REGISTER_PREPROCESS_ALGO(ClassName)`**
- **`REGISTER_INFER_ENGINE(ClassName)`**
- **`REGISTER_POSTPROCESS_ALGO(ClassName)`**

**示例：** 在你的 `my_postproc.cpp` 文件末尾添加：

```cpp
#include "ai_core/ai_core_registrar.hpp"
#include "my_postproc.hpp" // 包含你的插件类定义

// 假设你的后处理插件类名为 MyYoloPostproc
// 这行代码会将 "MyYoloPostproc" 这个字符串名称和你的类关联起来
REGISTER_POSTPROCESS_ALGO(MyYoloPostproc);
```
之后，在 `AlgoModuleTypes` 中设置 `postprocModule = "MyYoloPostproc"` 即可使用你的新插件。