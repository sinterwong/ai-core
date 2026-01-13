# AI Core
<p align="center">
  <img src="assets/icon/logo.jpeg" alt="ai-core Logo" width="500"> <br/>
</p>

<p align="center">
  一个高可扩展的 AI 算法库。
</p>

---
[English](README_EN.md) | [简体中文](README.md)
---

![Version](https://img.shields.io/badge/version-1.2.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)

**AI Core** 是一个现代化、高性能、可扩展的 C++ AI 推理框架。旨在简化 AI 模型在多种硬件后端上的部署流程，提供从数据预处理、模型推理到结果后处理的端到端解决方案。

## 核心特性

*   **📦 模块化流程 (Modular Pipeline):** 采用 **预处理 -> 推理 -> 后处理** 三段式流水线设计，每个阶段都可以独立实现和替换。
*   **🔌 可扩展插件系统 (Extensible Plugin System):** 基于工厂模式，您可以轻松地自定义预处理、推理引擎和后处理插件，并注册到框架中。
*   **🔒 类型安全的数据处理 (Type-Safe Data Handling):** 使用 `std::variant` 和自定义的 `TypedBuffer` 来管理不同类型的数据和参数，极大地提高了代码的健壮性和可维护性。
*   **🚀 硬件抽象与加速 (Hardware Abstraction & Acceleration):** 通过 `TypedBuffer` 统一管理 CPU 和 GPU 内存，无缝支持在不同设备间的数据流转与计算。
*   **✨ 简洁易用的高级 API (High-Level, Easy-to-Use API):** 提供了 `AlgoInference` 类作为统一入口，封装了复杂的底层调用，让开发者可以专注于业务逻辑。
*   **🔧 现代 C++ 设计 (Modern C++ Design):** 广泛采用 C++17/20 特性，保证了代码的高性能、高质量和安全性。


## 核心架构

AI Core 的核心是一个清晰的数据流管道。数据从输入 (`AlgoInput`) 开始，经过一系列处理模块，最终生成算法输出 (`AlgoOutput`)。

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
- **数据容器:** `TensorData` 是流水线内部的核心数据结构，用于在各个阶段之间传递张量数据。
- **插件:** 每个处理阶段（预处理、推理、后处理）都是一个可插拔的插件，通过字符串名称在运行时动态加载。

## 快速开始

### 环境要求

- C++20 兼容的编译器 (GCC 11+)
- CMake 3.15+
- (可选) CUDA Toolkit 11.x+
- (可选) OpenCV 4.x+


### 编译与安装

1.  **Clone the repository:**
    ```bash
    git clone --recurse-submodules https://github.com/sinterwong/ai-core.git
    cd ai-core
    mkdir -p 3rdparty/target/
    curl -L https://github.com/sinterwong/ai-core/releases/download/v1.1.1-alpha/dependency-Linux_x86_64.tgz -o dependency.tgz
    tar -xzf dependency.tgz -C 3rdparty/target/
    ```

2.  **Configure with CMake:**
    ```bash
    mkdir build && cd build
    cmake .. -DBUILD_AI_CORE_TESTS=ON -DBUILD_AI_CORE_EXAMPLES=ON -DWITH_ORT_ENGINE=ON -DWITH_NCNN_ENGINE=ON -DWITH_TRT_ENGINE=OFF
    ```
    *   Use `-DWITH_ORT_ENGINE=ON`, `-DWITH_NCNN_ENGINE=ON`, and `-DWITH_TRT_ENGINE=ON` to enable the respective inference engines.

3.  **Build the project:**
    ```bash
    cmake --build .
    ```
    
4. **Install:**
    ```bash
    cmake --install .
    ```

### 使用示例

暂时可以参考**tests**和**examples**目录中的内容。

## 文档

- **[框架设计与核心概念](./doc/Framework.md):** 解析 AI Core 的架构设计、核心组件和设计哲学。
- **[API 参考手册](./doc/API.md):** 详细介绍所有公开的类、函数和数据类型，并提供使用示例。
- **[快速入门](./doc/Quickstart.md):** (即将推出) 手把手教您如何使用 AI Core 完整地构建一个 AI 应用。
