# AI Core: 框架设计与核心概念

本文档旨在深入解析 **AI Core** 框架的架构设计、核心组件和设计哲学。阅读本文档将帮助您理解框架的工作原理，从而更高效地使用和扩展它。

- [1. 核心设计哲学](#1-核心设计哲学)
- [2. 架构概览：三段式流水线](#2-架构概览三段式流水线)
- [3. 核心数据流与关键数据结构](#3-核心数据流与关键数据结构)
  - [3.1 数据流转](#31-数据流转)
  - [3.2 TypedBuffer：类型感知的内存缓冲区](#32-typedbuffer类型感知的内存缓冲区)
  - [3.3 TensorData：张量数据集](#33-tensordata张量数据集)
  - [3.4 ParamCenter 与 std::variant：类型安全的参数化](#34-paramcenter-与-stdvariant类型安全的参数化)
  - [3.5 DataPacket 与 RuntimeContext：灵活的上下文传递](#35-datapacket-与-runtimecontext灵活的上下文传递)
- [4. 三大核心组件](#4-三大核心组件)
  - [4.1 组件一：预处理模块 (AlgoPreproc)](#41-组件一预处理模块-algopreproc)
  - [4.2 组件二：推理引擎 (AlgoInferEngine)](#42-组件二推理引擎-algoinferengine)
  - [4.3 组件三：后处理模块 (AlgoPostproc)](#43-组件三后处理模块-algopostproc)
- [5. 插件化核心：工厂与注册器系统](#5-插件化核心工厂与注册器系统)
- [6. 顶层协调器：AlgoInference](#6-顶层协调器algoinference)
- [7. 总结](#7-总结)

---

## 1. 核心设计哲学

AI Core 框架的设计根植于以下几个核心原则：

- **分离关注点 (Separation of Concerns):** 严格划分数据预处理、模型推理和业务逻辑后处理的职责。每个部分都应是内聚的、独立的模块。
- **面向接口编程 (Programming to an Interface):** 上层模块不依赖于下层模块的具体实现，而是依赖于抽象接口。这使得框架的任何部分都可以被轻松替换或扩展。
- **类型安全优于一切 (Type Safety Above All):** 尽最大努力在编译期捕获类型错误，避免使用 `void*` 等不安全的裸指针。通过现代 C++ 特性（`std::variant`, `std::any`, `template`）在保证灵活性的同时提供强大的类型检查。
- **显式管理资源 (Explicit Resource Management):** 模块的生命周期（初始化、执行、终止）被明确定义和管理，确保资源（如 GPU 显存）被正确申请和释放。

## 2. 架构概览：三段式流水线

AI Core 的宏观架构是一个经典的三段式流水线（Pipeline）。这种设计将一个复杂的端到端AI任务分解为三个逻辑上独立的阶段：

1.  **预处理 (Pre-processing):** 负责将原始输入（如图像、文本）转换为模型所需的标准化张量（Tensor）格式。
2.  **推理 (Inference):** 负责将预处理后的张量输入给AI模型，并执行前向传播，得到模型的原始输出张量。
3.  **后处理 (Post-processing):** 负责将模型的原始输出张量解析为对业务有意义的、结构化的结果（如检测框、分类标签）。

```
+-----------+     +-----------------+     +-----------------+
|           |     |                 |     |                 |
| AlgoInput |---->| Preproc Plugin  |---->|  TensorData (in)|
|           |     | (e.g. FrameProc)|     |                 |
+-----------+     +-----------------+     +-----------------+
                                                   |
                                                   v
                                           +-----------------+
                                           |                 |
                                           | Inference Engine|
                                           | (e.g., TensorRT)|
                                           +-----------------+
                                                    |
                                                    v       
+-----------+     +------------------+     +------------------+
|           |     |                  |     |                  |
| AlgoOutput|<----| Postproc Plugin  |<----| TensorData (out) |
|           |     | (e.g. YOLO_DET ) |     |                  |
+-----------+     +------------------+     +------------------+                                         
                                         
```

## 3. 核心数据流与关键数据结构

理解数据如何在流水线中流动是理解整个框架的关键。

### 3.1 数据流转

- **`AlgoInput`**: 流水线的起点，代表最原始的输入数据。
- **`TensorData`**: 流水线内部的“血液”，用于在预处理、推理和后处理阶段之间传递张量数据。
- **`AlgoOutput`**: 流水线的终点，代表经过解析的、对用户友好的算法结果。

### 3.2 `TypedBuffer`：类型感知的内存缓冲区

`TypedBuffer` 是框架中最重要的底层数据结构。它不是一个简单的字节数组，而是一个智能的、类型感知的缓冲区，具备以下特点：

- **位置无关性:** 它可以管理 **CPU 内存** (`std::vector<uint8_t>`) 或 **GPU 显存** (`DeviceBufferImpl`)。上层代码无需关心数据具体存储在哪里，`TypedBuffer` 屏蔽了硬件细节。
- **类型感知:** 它存储了数据的类型 `DataType` (如 `FLOAT32`, `INT8`) 和元素数量。这使得在访问数据时可以进行类型和大小的检查，避免了常见的内存错误。
- **所有权管理:** 它可以完全拥有和管理内存（例如，通过 `createFromGpu` 创建），也可以仅引用外部传入的内存（通过 `setGpuDataReference`），提供了灵活的内存管理策略。

### 3.3 `TensorData`：张量数据集

`TensorData` 是一个容器，代表模型一次推理所需的所有输入或输出张量。它本质上是一个从张量名称（`std::string`）到 `TypedBuffer` 的映射：

```cpp
struct TensorData {
  std::map<std::string, TypedBuffer> datas;
  std::map<std::string, std::vector<int>> shapes;
};
```

这种设计使得按名称访问特定张量变得非常方便，与现代推理引擎（如 ONNX Runtime, TensorRT）的输入输出绑定方式完全契合。

### 3.4 `ParamCenter` 与 `std::variant`：类型安全的参数化

为了在不牺牲类型安全的前提下处理多样的输入/输出和参数类型，框架引入了 `ParamCenter` 模板类，它巧妙地包装了 C++17 的 `std::variant`。

- `AlgoInput`: 可以是 `FrameInput` (单帧图像)，也可以是 `FrameInputWithMask` (带掩码的图像)。
- `AlgoOutput`: 可以是 `DetRet` (检测结果)，也可以是 `ClsRet` (分类结果)。
- `AlgoPreprocParams` / `AlgoPostprocParams`: 同理，可以是针对不同任务的特定参数结构体。

使用 `ParamCenter` 的好处是：
1.  **静态类型检查:** 编译器可以检查你是否在 `AlgoInput` 中存入了合法的类型。
2.  **避免虚函数开销:** 相比于使用继承和虚函数，`std::variant` 通常有更好的性能。
3.  **清晰的接口:** `infer` 函数的签名是固定的，而无需为每一种输入/输出类型重载。

```cpp
// 设置参数 (类型安全)
AlgoInput input;
input.setParams(FrameInput{...});

// 获取参数 (类型安全)
if (auto* frame_input = input.getParams<FrameInput>()) {
    // 安全地使用 frame_input
}
```

### 3.5 `DataPacket` 与 `RuntimeContext`：灵活的上下文传递

`DataPacket` 是一个基于 `std::map<std::string, std::any>` 的通用数据包。它被用在两个地方：

1.  **`AlgoConstructParams`**: 用于向插件的构造函数传递配置信息。
2.  **`RuntimeContext`**: 作为一个可选的上下文对象，在 `infer` 调用的整个生命周期中传递。它非常适合传递一些“元数据”，例如：预处理模块可以将图像的原始尺寸、缩放比例、填充大小等信息放入 `RuntimeContext`，后处理模块再从中取出这些信息来将检测框坐标还原到原始图像尺度。

---

## 4. 三大核心组件

这三个组件是构成推理流水线的具体执行者。它们本身是高级封装，内部通过工厂模式创建并持有一个具体的插件实现。**对于使用者来说，也可以灵活选择是否拆开使用三大组件**。

### 4.1 组件一：预处理模块 (`AlgoPreproc`)

- **职责:** 实现流水线的第一个阶段。它负责将 `AlgoInput` 中的原始数据（如 `cv::Mat`）转换为模型输入所需的 `TensorData`。
- **接口:** 内部持有一个 `IPreprocssPlugin` 接口的共享指针。所有具体的预处理算法都必须实现这个接口。
- **输入:** `const AlgoInput&`, `const AlgoPreprocParams&`
- **输出:** `TensorData&`
- **工作模式:** `AlgoPreproc` 的构造函数接收一个字符串模块名。在 `initialize()` 期间，它会请求 `PreprocFactory` 使用这个名称来创建一个具体的插件实例。随后的 `process()` 调用都会被转发到这个插件实例上。

### 4.2 组件二：推理引擎 (`AlgoInferEngine`)

- **职责:** 实现流水线的核心——模型推理。它接收 `TensorData` 作为输入，调用底层推理后端（如 TensorRT, ONNX Runtime）执行计算，并返回包含模型原始输出的 `TensorData`。
- **接口:** 内部持有一个 `IInferEnginePlugin` 接口的共享指针。所有对不同推理后端的封装（如 `OrtAlgoInference`, `TrtAlgoInference`）都必须实现此接口。
- **输入:** `const TensorData&`
- **输出:** `TensorData&`
- **工作模式:** 与 `AlgoPreproc` 类似，`AlgoInferEngine` 在构造时接收模块名和推理配置 `AlgoInferParams`（包含模型路径、设备类型等）。在 `initialize()` 时，它会创建并初始化底层的推理引擎插件。`infer()` 方法负责将输入 `TensorData` 中的 `TypedBuffer` 绑定到引擎的输入端，执行推理，然后将结果填充到输出 `TensorData` 中。

### 4.3 组件三：后处理模块 (`AlgoPostproc`)

- **职责:** 实现流水线的最后阶段。它负责将 `AlgoInferEngine` 输出的原始 `TensorData`（通常是高维浮点数组）解析成对业务有意义、易于使用的结构化数据，并填充到 `AlgoOutput` 中。
- **接口:** 内部持有一个 `IPostprocssPlugin` 接口的共享指针。所有具体的后处理算法都必须实现这个接口。
- **输入:** `const TensorData&`, `const AlgoPostprocParams&`
- **输出:** `AlgoOutput&`
- **工作模式:** `AlgoPostproc` 的工作模式与 `AlgoPreproc` 完全一致。它根据模块名从 `PostprocFactory` 获取插件实例，并将处理任务委托给该实例。

## 5. 插件化核心：工厂与注册器系统

框架的可扩展性源于其**插件化设计**，而这一设计的基石是 **工厂模式 (Factory Pattern)** 和 **自注册机制 (Self-Registration)**。

- **接口 (`I...Plugin`):** 定义了每种插件（预处理、推理、后处理）必须遵守的“契约”。
- **工厂 (`Factory<T>`):** 每个插件类型都有一个对应的单例工厂（如 `PreprocFactory`）。工厂内部维护一个从字符串名称到插件创建函数的映射表。
- **注册器 (`REGISTER_...` 宏):** 开发者在实现一个新的插件后，只需在 `.cpp` 文件末尾使用相应的宏（如 `REGISTER_POSTPROCESS_ALGO(MyCustomPostProc)`），即可在程序启动时自动将该插件的创建函数注册到对应的工厂中。

这个系统使得添加一个全新的算法或推理后端变得极其简单，**完全无需修改框架的任何核心代码**。

```cpp
// in my_custom_postproc.cpp

class MyCustomPostProc : public ai_core::dnn::IPostprocssPlugin {
    // ... implement process() and batchProcess() ...
};

// 使用宏将新插件注册到工厂
REGISTER_POSTPROCESS_ALGO(MyCustomPostProc);
```

## 6. 顶层协调器：`AlgoInference`

如果说三大组件是流水线上的工人，那么 `AlgoInference` 就是管理整个流水线的“工头”。它扮演着**外观（Facade）**和**协调器（Orchestrator）**的角色。

- **职责:**
    1.  **组装流水线:** 在构造时接收 `AlgoModuleTypes`（包含三个组件的模块名）和 `AlgoInferParams`。
    2.  **管理生命周期:** 在其 `initialize()` 方法中，它会依次创建并初始化 `AlgoPreproc`, `AlgoInferEngine`, `AlgoPostproc` 三个组件。`terminate()` 方法则负责逆序销毁它们。
    3.  **编排数据流:** 它的 `infer()` 方法负责按顺序调用三大组件的 `process` 或 `infer` 方法，并管理 `TensorData` 在它们之间的传递。
- **对用户:** `AlgoInference` 是与框架交互的主要入口点。它向用户隐藏了内部组件的复杂交互和生命周期管理，提供了一个干净、统一的接口。

## 7. 总结

AI Core 框架通过分层设计、面向接口编程和强大的类型安全机制，构建了一个健壮、灵活且可扩展的推理平台。

- **分层:** `AlgoInference` (顶层) -> 三大组件 (中层) -> 插件接口与实现 (底层)。
- **解耦:** 工厂模式和插件化设计使得各组件之间、以及框架与具体实现之间高度解耦。
- **安全与高效:** `TypedBuffer` 和 `ParamCenter` 等数据结构确保了数据在流水线中安全、高效地流动，同时无缝支持跨 CPU/GPU 的异构计算。

理解这些核心概念，将使您能够充分利用 AI Core 框架的强大功能，并根据业务需求对其进行定制和扩展。
