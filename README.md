# AI Core
<p align="center">
  <img src="assets/icon/logo.jpeg" alt="ai-core Logo" width="500"> <br/>
</p>

<p align="center">
  A highly scalable deep learning algorithm computing library.
</p>

AI Core is a powerful and extensible C++ library for managing and executing AI inference algorithms. Designed with a modular architecture, it allows developers to seamlessly integrate and switch between different inference engines like ONNX Runtime, NCNN, and TensorRT.

## Features

*   **Multiple Inference Engines:** Supports ONNX Runtime, NCNN, and TensorRT.
*   **Modular Design:** Easily extend the library by adding new algorithms and pre/post-processing modules.

## Core Components

AI Core is built around three main components that form the foundation of the inference pipeline: `AlgoPreproc`, `AlgoInferEngine`, and `AlgoPostproc`. These components work together to handle the entire process, from data preparation to final output, allowing for a flexible and modular approach to building and running AI models.

-   **`AlgoPreproc`**: Responsible for all data preprocessing tasks. This component takes raw input, such as images or other data, and transforms it into the format required by the inference engine. This can include resizing, normalization, and other transformations.

-   **`AlgoInferEngine`**: The core of the inference process. This component takes the preprocessed data and runs it through the deep learning model using a specified backend, such as ONNX Runtime, NCNN, or TensorRT. It manages the model loading, execution, and data transfer between the host and the device.

-   **`AlgoPostproc`**: Handles the post-processing of the model's output. This component takes the raw output from the inference engine and transforms it into a human-readable and usable format, such as bounding boxes for object detection or masks for segmentation.

These components can be used individually for fine-grained control over the inference pipeline or orchestrated together to create a seamless end-to-end workflow. The following sections provide a more detailed look at each of these components.

## Getting Started

### Prerequisites

*   C++20 compliant compiler (e.g., GCC 9+, Clang 9+).
*   CMake (3.16 or higher).

### Dependencies

AI Core requires the following dependencies:

*   **ONNX Runtime:** For running ONNX models if you use.
*   **NCNN:** For running NCNN models if you use.
*   **TensorRT:** For running TensorRT engines if you use.
*   **OpenCV:** For image pre-processing.
*   **[logger](https://github.com/sinterwong/logger.git):** For logging.
*   **[encryption-tool](https://github.com/sinterwong/encryption-tool.git):** For encryption file.
*   **nlohmann-json:** For parsing JSON configuration files.

The project uses CMake to manage dependencies. You can either install them on your system or use a dependency manager like vcpkg or Conan.

### Building

1.  **Clone the repository:**
    ```bash
    git clone --recurse-submodules https://github.com/sinterwong/ai-core.git
    cd ai-core
    ```
    If you have already cloned the repository without the submodules, you can initialize them:
    ```bash
    git submodule update --init --recursive
    ```

2.  **Configure with CMake:**
    ```bash
    mkdir build && cd build
    cmake .. -DBUILD_AI_CORE_TESTS=ON -DWITH_ORT_ENGINE=ON -DWITH_NCNN_ENGINE=ON -DWITH_TRT_ENGINE=OFF
    ```
    *   Use `-DWITH_ORT_ENGINE=ON`, `-DWITH_NCNN_ENGINE=ON`, and `-DWITH_TRT_ENGINE=ON` to enable the respective inference engines.

3.  **Build the project:**
    ```bash
    cmake --build .
    ```

### Installation

To install the library, run the following command from the `build` directory:

```bash
cmake --install .
```

### Higher-Level Abstractions

For convenience, AI Core also provides higher-level abstractions like `AlgoInference`. These classes encapsulate the entire pipeline and are useful when you don't need fine-grained control over each step.

-   **`AlgoInference`**: Wraps the three-stage pipeline (`AlgoPreproc`, `AlgoInferEngine`, `AlgoPostproc`) into a single class for easier use.

## API Documentation

For a detailed description of the public API, including classes, methods, and data structures, please see the [API Documentation](doc/API.md).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
