# AI Core

<p align="center">
  <img src="assets/icon/logo.jpeg" alt="ai-core Logo" width="500"><br/>
</p>

<p align="center">A C++ AI inference framework</p>

[English](README_EN.md) | [简体中文](README.md)

![Version](https://img.shields.io/badge/version-1.2.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)

AI Core is a C++ library for running AI models on multiple backends (ONNX Runtime, NCNN, TensorRT). A pipeline is built from three pluggable stages — preprocessor, inference engine, postprocessor — that you register by name and assemble at runtime.

## Pipeline

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

`TensorData` is the named-tensor map that moves between stages. `AlgoInput` and `AlgoOutput` are the user-facing types at the ends of the pipeline.

## Build

### Requirements

- A C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+)
- CMake 3.18+
- OpenCV 4.x
- ONNX Runtime (enabled by default)
- Optional: NCNN, TensorRT, CUDA Toolkit

### Clone and build

```bash
git clone --recurse-submodules https://github.com/sinterwong/ai-core.git
cd ai-core
mkdir -p 3rdparty/target/

# Pre-built third-party dependencies
curl -L https://github.com/sinterwong/ai-core/releases/download/v1.1.1-alpha/dependency-Linux_x86_64.tgz -o dependency.tgz
tar -xzf dependency.tgz -C 3rdparty/target/

mkdir build && cd build
cmake .. -DBUILD_AI_CORE_EXAMPLES=ON -DBUILD_AI_CORE_TESTS=ON \
         -DWITH_ORT_ENGINE=ON -DWITH_NCNN_ENGINE=ON -DWITH_TRT_ENGINE=OFF

cmake --build . -j
cmake --install .
```

CMake options:

| Option | Default | Description |
| --- | --- | --- |
| `BUILD_AI_CORE_TESTS` | OFF | Build unit tests |
| `BUILD_AI_CORE_BENCHMARKS` | OFF | Build benchmarks |
| `BUILD_AI_CORE_EXAMPLES` | OFF | Build examples |
| `WITH_ORT_ENGINE` | ON | ONNX Runtime backend |
| `WITH_NCNN_ENGINE` | OFF | NCNN backend |
| `WITH_TRT_ENGINE` | OFF | TensorRT backend |

## Usage

`AlgoInference` is the pipeline entry point. Pass it the three plugin names and the inference parameters:

```cpp
#include "ai_core/algo_inference.hpp"
#include "ai_core/algo_types.hpp"

using namespace ai_core;

AlgoModuleTypes modules{
    "FramePreprocess",     // preprocessor
    "OrtAlgoInference",    // backend
    "AnchorDetPostproc"    // postprocessor
};

AlgoInferParams params;
params.name = "yolov11";
params.model_path = "models/yolov11.onnx";
params.device_type = DeviceType::CPU;
params.data_type = DataType::FLOAT32;

dnn::AlgoInference algo(modules, params);
algo.initialize();

AlgoInput input;
input.setParams(FrameInput{
    std::make_shared<cv::Mat>(cv::imread("test.jpg")),
    std::make_shared<cv::Rect>(0, 0, 0, 0)
});

AlgoPreprocParams preproc_params;
FramePreprocessArg arg;
arg.model_input_shape = {640, 640, 3};
arg.mean_vals = {0.f, 0.f, 0.f};
arg.norm_vals = {1.f, 1.f, 1.f};
arg.hwc2chw = true;
arg.data_type = DataType::FLOAT32;
preproc_params.setParams(arg);

AlgoPostprocParams postproc_params;
AnchorDetParams det_arg;
det_arg.algo_type = AnchorDetParams::AlgoType::YoloDetV11;
det_arg.cond_thre = 0.25f;
det_arg.nms_thre = 0.45f;
det_arg.output_names = {"output0"};
postproc_params.setParams(det_arg);

AlgoOutput output;
if (algo.infer(input, preproc_params, postproc_params, output)
    != InferErrorCode::SUCCESS) {
    // handle error
}

if (auto* det = output.getParams<DetRet>()) {
    for (const auto& box : det->bboxes) {
        // ...
    }
}

algo.terminate();
```

More complete samples are in `examples/generic_image_infer.cpp`, `examples/ocr/`, and `tests/`.

## Documentation

- [doc/Framework.md](doc/Framework.md) — framework structure and design
- [doc/API.md](doc/API.md) — public API reference

## License

MIT
