# AI Core
<p align="center">
  <img src="assets/icon/logo.jpeg" alt="ai-core Logo" width="500"> <br/>
</p>

<p align="center">
  A highly extensible AI algorithm library.
</p>

![Version](https://img.shields.io/badge/version-1.1.2-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![C++ Standard](https://img.shields.io/badge/C++-17-blue.svg)

**AI Core** is a modern, high-performance, and extensible C++ AI inference framework. It is designed to streamline the deployment of AI models across multiple hardware backends, offering an end-to-end solution that covers data preprocessing, model inference, and result post-processing.

## Core Features

*   **📦 Modular Pipeline:** Adopts a three-stage pipeline design: **Preprocessing -> Inference -> Post-processing**. Each stage can be implemented and replaced independently.
*   **🔌 Extensible Plugin System:** Based on the factory pattern, you can easily customize and register preprocessing, inference engine, and post-processing plugins within the framework.
*   **🔒 Type-Safe Data Handling:** Utilizes `std::variant` and a custom `TypedBuffer` to manage different types of data and parameters, significantly enhancing code robustness and maintainability.
*   **🚀 Hardware Abstraction & Acceleration:** Manages CPU and GPU memory uniformly through `TypedBuffer`, enabling seamless data transfer and computation across different devices.
*   **✨ High-Level, Easy-to-Use API:** Provides the `AlgoInference` class as a unified entry point, encapsulating complex low-level calls and allowing developers to focus on their application logic.
*   **🔧 Modern C++ Design:** Makes extensive use of C++17/20 features to ensure high performance, quality, and code safety.


## Core Architecture

The core of AI Core is a clear data flow pipeline. Data starts from the input (`AlgoInput`), passes through a series of processing modules, and finally generates the algorithm output (`AlgoOutput`).

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
- **Data Container:** `TensorData` is the core data structure within the pipeline, used to pass tensor data between stages.
- **Plugins:** Each processing stage (preprocessing, inference, post-processing) is a pluggable component, dynamically loaded at runtime using its string name.

## Quick Start

### Prerequisites

- A C++20 compatible compiler (e.g., GCC 11+)
- CMake 3.15+
- (Optional) CUDA Toolkit 11.x+
- (Optional) OpenCV 4.x+


### Build and Install

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

### Usage Example

For now, please refer to the content in the **tests** and **examples** directories.

## Documentation

- **[Framework Design and Core Concepts](./doc/Framework.md):** An explanation of AI Core's architecture, core components, and design philosophy.
- **[API Reference](./doc/API.md):** A detailed guide to all public classes, functions, and data types, including usage examples.
- **[Quick Start Guide](./doc/Quickstart.md):** (Coming soon) A step-by-step tutorial on building a complete AI application with AI Core.
