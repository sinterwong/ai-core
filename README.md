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
*   **Configuration-based:** Easily use JSON files to define and configure your AI algorithms.

## Core Components

AI Core is built around three main components that form the foundation of the inference pipeline: `AlgoPreproc`, `AlgoInferEngine`, and `AlgoPostproc`. These components work together to handle the entire process, from data preparation to final output, allowing for a flexible and modular approach to building and running AI models.

-   **`AlgoPreproc`**: Responsible for all data preprocessing tasks. This component takes raw input, such as images or other data, and transforms it into the format required by the inference engine. This can include resizing, normalization, and other transformations.

-   **`AlgoInferEngine`**: The core of the inference process. This component takes the preprocessed data and runs it through the deep learning model using a specified backend, such as ONNX Runtime, NCNN, or TensorRT. It manages the model loading, execution, and data transfer between the host and the device.

-   **`AlgoPostproc`**: Handles the post-processing of the model's output. This component takes the raw output from the inference engine and transforms it into a human-readable and usable format, such as bounding boxes for object detection or masks for segmentation.

These components can be used individually for fine-grained control over the inference pipeline or orchestrated together to create a seamless end-to-end workflow. The following sections provide a more detailed look at each of these components.

### `AlgoPreproc`: Data Preprocessing

The `AlgoPreproc` class is responsible for preparing raw input data for the inference engine. It takes an `AlgoInput` object, which can contain various data types like images or sensor data, and applies a series of transformations defined in `AlgoPreprocParams`. The output is a `TensorData` object, which is ready to be consumed by the `AlgoInferEngine`.

Here's how to use `AlgoPreproc` to preprocess an image for a typical computer vision model:

```cpp
#include "ai_core/algo_preproc.hpp"
#include "ai_core/algo_data_types.hpp"
#include <opencv2/opencv.hpp>

// 1. Initialize the preprocessor with a specific module name
auto preproc = std::make_shared<ai_core::dnn::AlgoPreproc>("FramePreprocess");
preproc->initialize();

// 2. Prepare the input data
cv::Mat image = cv::imread("path/to/your/image.jpg");
auto algo_input = std::make_shared<ai_core::dnn::AlgoInput>();
algo_input->setParams(image);

// 3. Set up preprocessing parameters
ai_core::dnn::AlgoPreprocParams preproc_params;
// ... configure parameters such as resize dimensions, normalization, etc.

// 4. Create a TensorData object for the output
ai_core::dnn::TensorData model_input;

// 5. Run the preprocessing
preproc->process(*algo_input, preproc_params, model_input);

// 'model_input' now contains the preprocessed data ready for inference.
```

### `AlgoInferEngine`: Core Inference Execution

The `AlgoInferEngine` is the heart of the AI Core library, responsible for executing the deep learning model. It abstracts the complexities of different inference backends, providing a unified interface for running inference. You can choose from various supported backends like ONNX Runtime, NCNN, or TensorRT by specifying the module name during initialization.

Here’s an example of how to initialize and use the `AlgoInferEngine`:

```cpp
#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/infer_params_types.hpp"

// 1. Define inference parameters, such as the model path and device type
ai_core::dnn::AlgoInferParams infer_params;
infer_params.modelPath = "path/to/your/model.onnx";
infer_params.deviceType = ai_core::dnn::DeviceType::CPU;

// 2. Create an AlgoInferEngine instance with a specific backend
auto engine = std::make_shared<ai_core::dnn::AlgoInferEngine>("OrtAlgoInference", infer_params);
if (engine->initialize() != ai_core::dnn::InferErrorCode::SUCCESS) {
    // Handle initialization failure
    return -1;
}

// 3. Prepare the input tensor (e.g., from AlgoPreproc)
ai_core::dnn::TensorData model_input;
// ... populate model_input ...

// 4. Create a TensorData object for the output
ai_core::dnn::TensorData model_output;

// 5. Run inference
engine->infer(model_input, model_output);

// 'model_output' now contains the raw output from the model.
```

### `AlgoPostproc`: Processing Model Output

After the `AlgoInferEngine` produces the raw output, the `AlgoPostproc` class is used to transform this data into a structured and meaningful format. It takes the `TensorData` from the inference stage and applies post-processing logic, such as decoding bounding boxes, applying non-maximum suppression, or generating segmentation masks.

Here's how to use `AlgoPostproc` to process the output of an object detection model:

```cpp
#include "ai_core/algo_postproc.hpp"
#include "ai_core/algo_data_types.hpp"

// 1. Initialize the post-processor with a specific module name
auto postproc = std::make_shared<ai_core::dnn::AlgoPostproc>("CVGenericPostproc");
postproc->initialize();

// 2. Prepare the model output (from AlgoInferEngine)
ai_core::dnn::TensorData model_output;
// ... populate model_output ...

// 3. Set up post-processing parameters
ai_core::dnn::AlgoPostprocParams postproc_params;
// ... configure parameters like confidence thresholds, NMS values, etc.

// 4. Create an AlgoOutput object to store the final results
ai_core::dnn::AlgoOutput algo_output;

// 5. Run the post-processing
postproc->process(model_output, algo_output, postproc_params);

// 'algo_output' now contains the processed results, such as bounding boxes and labels.
```

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

## Usage: End-to-End Inference Pipeline

While the `AlgoPreproc`, `AlgoInferEngine`, and `AlgoPostproc` components can be used individually, they are most powerful when combined to create a complete end-to-end inference pipeline. This approach allows you to build flexible and high-performance AI applications with clear separation of concerns.

The following example demonstrates how to combine these three components to perform object detection on an image.

```cpp
#include "ai_core/algo_preproc.hpp"
#include "ai_core/algo_infer_engine.hpp"
#include "ai_core/algo_postproc.hpp"
#include "ai_core/algo_data_types.hpp"
#include <opencv2/opencv.hpp>

int main() {
    // 1. Initialize Preprocessing
    auto preproc = std::make_shared<ai_core::dnn::AlgoPreproc>("FramePreprocess");
    preproc->initialize();

    // 2. Initialize Inference Engine
    ai_core::dnn::AlgoInferParams infer_params;
    infer_params.modelPath = "path/to/your/model.onnx";
    infer_params.deviceType = ai_core::dnn::DeviceType::CPU;
    auto engine = std::make_shared<ai_core::dnn::AlgoInferEngine>("OrtAlgoInference", infer_params);
    engine->initialize();

    // 3. Initialize Post-processing
    auto postproc = std::make_shared<ai_core::dnn::AlgoPostproc>("CVGenericPostproc");
    postproc->initialize();

    // 4. Load and Prepare Input Data
    cv::Mat image = cv::imread("path/to/your/image.jpg");
    auto algo_input = std::make_shared<ai_core::dnn::AlgoInput>();
    algo_input->setParams(image);

    // 5. Define Preprocessing and Post-processing Parameters
    ai_core::dnn::AlgoPreprocParams preproc_params;
    // ... configure preprocessing parameters ...
    ai_core::dnn::AlgoPostprocParams postproc_params;
    // ... configure post-processing parameters ...

    // 6. Run the Full Pipeline
    ai_core::dnn::TensorData model_input;
    preproc->process(*algo_input, preproc_params, model_input);

    ai_core::dnn::TensorData model_output;
    engine->infer(model_input, model_output);

    ai_core::dnn::AlgoOutput algo_output;
    postproc->process(model_output, algo_output, postproc_params);

    // 7. Use the Final Output
    // ... process the results in 'algo_output' ...

    return 0;
}
```

### Higher-Level Abstractions

For convenience, AI Core also provides higher-level abstractions like `AlgoInference` and `AlgoManager`. These classes encapsulate the entire pipeline and are useful when you don't need fine-grained control over each step.

-   **`AlgoInference`**: Wraps the three-stage pipeline (`AlgoPreproc`, `AlgoInferEngine`, `AlgoPostproc`) into a single class for easier use.
-   **`AlgoManager`**: Manages multiple `AlgoInference` instances and allows you to run them by name, based on a JSON configuration file.

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
                "postproc": "AnchorDetPostproc"
            },
            "preprocParams": {
                "inputShape": {
                    "w": 640,
                    "h": 640,
                    "c": 3
                },
                "mean": [
                    0.0,
                    0.0,
                    0.0
                ],
                "std": [
                    255.0,
                    255.0,
                    255.0
                ],
                "pad": [
                    0,
                    0,
                    0
                ],
                "isEqualScale": true,
                "needResize": true,
                "dataType": 1,
                "hwc2chw": true,
                "inputNames": [
                    "images"
                ],
                "preprocTaskType": 0
            },
            "inferParams": {
                "name": "yolo-det-80c-001",
                "modelPath": "models/yolov11n-fp16.onnx",
                "deviceType": 0,
                "dataType": 1,
                "needDecrypt": false
            },
            "postprocParams": {
                "detAlogType": 0,
                "condThre": 0.5,
                "nmsThre": 0.45,
                "outputNames": [
                    "output0"
                ]
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
