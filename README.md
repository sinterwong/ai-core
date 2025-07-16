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
*   **Configuration-based:** Use JSON files to define and configure your AI algorithms.

## Getting Started

### Prerequisites

*   C++17 compliant compiler (e.g., GCC 9+, Clang 9+).
*   CMake (3.16 or higher).
*   Git.

### Dependencies

AI Core requires the following dependencies:

*   **ONNX Runtime:** For running ONNX models if you use.
*   **NCNN:** For running NCNN models if you use.
*   **TensorRT:** For running TensorRT engines if you use.
*   **OpenCV:** For image pre-processing.
*   **[logger](https://github.com/sinterwong/logger.git):** For logging.
*   **[encryption-tool](https://github.com/sinterwong/encryption-tool.git):** For encryption file.
*   **[cpp-common-utils](https://github.com/sinterwong/cpp-common-utils.git):** For the The data structures and tools used for building the project.
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

## Usage

### `AlgoInference`: The Core of Execution

The `AlgoInference` class is the workhorse of AI Core. It encapsulates the entire inference pipeline for a single algorithm, including pre-processing, inference, and post-processing. You can use it directly for more fine-grained control over the inference process.

Here's an example of how to use `AlgoInference` for a YOLOv11 object detection model:

```cpp
#include "ai_core/algo_infer.hpp"
#include "ai_core/algo_data_types.hpp"
#include <opencv2/opencv.hpp>

int main() {
    // 1. Define the algorithm modules and parameters
    ai_core::dnn::AlgoModuleTypes module_types;
    module_types.preproc = "FramePreprocess";
    module_types.infer = "OrtAlgoInference";
    module_types.postproc = "Yolov11Det";

    ai_core::dnn::AlgoInferParams infer_params;
    infer_params.modelPath = "models/yolov11n-fp16.onnx";
    infer_params.deviceType = ai_core::dnn::DeviceType::CPU;

    // 2. Create an AlgoInference instance
    auto algo = std::make_shared<ai_core::dnn::AlgoInference>(module_types, infer_params);
    if (algo->initialize() != ai_core::dnn::InferErrorCode::SUCCESS) {
        return -1;
    }

    // 3. Create input data structures
    ai_core::dnn::AlgoInput input;
    // ... set input data ...

    // 4. Create output data structures
    ai_core::dnn::AlgoOutput output;

    // 5. Define pre-processing and post-processing parameters
    ai_core::dnn::AlgoPreprocParams preproc_params;
    // ... set pre-processing parameters ...

    ai_core::dnn::AlgoPostprocParams postproc_params;
    // ... set post-processing parameters ...

    // 6. Run inference
    algo->infer(input, preproc_params, output, postproc_params);

    // 7. Process the output
    // ...

    return 0;
}
```

### `AlgoManager`: Simplified Algorithm Management

The `AlgoManager` class provides a higher-level interface for managing multiple algorithms. It loads algorithm configurations from a JSON file and provides a simple way to run inference by name.

### Configuration

AI Core uses JSON files to configure algorithms. Here's an example configuration for a YOLOv11 object detection model using ONNX Runtime:

```json
{
    "algorithms": [
        {
            "name": "yolo-det-80c",
            "types": {
                "preproc": "FramePreprocess",
                "infer": "OrtAlgoInference",
                "postproc": "Yolov11Det"
            },
            "preprocParams": {
                "inputShape": { "w": 640, "h": 640, "c": 3 },
                "mean": [ 0.0, 0.0, 0.0 ],
                "std": [ 255.0, 255.0, 255.0 ],
                "isEqualScale": true,
                "hwc2chw": true
            },
            "inferParams": {
                "modelPath": "models/yolov11n-fp16.onnx",
                "deviceType": 0
            },
            "postprocParams": {
                "condThre": 0.5,
                "nmsThre": 0.45
            }
        }
    ]
}
```

## Running Tests

To run the tests, first build the project with tests enabled:

```bash
cmake .. -DBUILD_AI_CORE_TESTS=ON
cmake --build .
```

Then, from the `build` directory, run:

```bash
./bin/ai_core_tests
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
