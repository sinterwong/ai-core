# AI Core

<p align="center">
  <img src="assets/icon/logo.jpeg" alt="ai-core Logo" width="500"><br/>
</p>

<p align="center">A C++ AI inference framework</p>

[English](README_EN.md) | [简体中文](README.md)

![Version](https://img.shields.io/badge/version-2.1.0-blue)
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
- Core library: a C++20 compiler, with no third-party runtime dependency
- Bundled OpenCV pre/post-process plugins: OpenCV 4.x
- Optional inference plugins: ONNX Runtime, NCNN, TensorRT/CUDA

Only the core is built by default. Enable repository-maintained plugins with
`AI_CORE_BUILD_BUNDLED_PLUGINS=ON`; inference plugins remain controlled by
`WITH_ORT_ENGINE`, `WITH_NCNN_ENGINE`, and `WITH_TRT_ENGINE`. In-tree and
out-of-tree plugins are both loaded through `PluginManager` and are never
folded into `libai_core.so`.

### Clone and build

The bootstrap script is the short path: dependencies, configure, build, install
and tests in one command.

```bash
git clone --recurse-submodules https://github.com/sinterwong/ai-core.git
cd ai-core
sudo apt-get install -y ninja-build
scripts/bootstrap.sh
```

To do it by hand:

```bash
# ONNX Runtime currently comes from its official release. OpenCV 4.10.0 is a
# pinned source submodule, trimmed by 3rdparty/CMakeLists.txt and linked
# statically into only the plugins that need it.
ORT_VERSION=1.20.1
mkdir -p 3rdparty/target/Linux_x86_64
curl -fL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
  -o /tmp/ort.tgz
tar -xzf /tmp/ort.tgz -C /tmp
mv "/tmp/onnxruntime-linux-x64-${ORT_VERSION}" 3rdparty/target/Linux_x86_64/onnxruntime

# The 1.20.x tarball ships lib/ + include/, but its own cmake export points at
# lib64/ and include/onnxruntime. Without this, find_package resolves paths
# that do not exist.
ORT_CMAKE=3rdparty/target/Linux_x86_64/onnxruntime/lib/cmake/onnxruntime
sed -i 's#/lib64/#/lib/#g' $ORT_CMAKE/onnxruntimeTargets-release.cmake
sed -i 's#/include/onnxruntime"#/include"#g' $ORT_CMAKE/onnxruntimeTargets.cmake

cmake -B build -DBUILD_AI_CORE_EXAMPLES=ON -DBUILD_AI_CORE_TESTS=ON \
      -DAI_CORE_BUILD_BUNDLED_PLUGINS=ON -DWITH_ORT_ENGINE=ON \
      -DWITH_TRT_ENGINE=OFF
cmake --build build -j
cmake --install build
```

CMake options:

| Option | Default | Description |
| --- | --- | --- |
| `BUILD_AI_CORE_TESTS` | OFF | Build unit tests |
| `BUILD_AI_CORE_BENCHMARKS` | OFF | Build benchmarks |
| `BUILD_AI_CORE_EXAMPLES` | OFF | Build examples |
| `AI_CORE_BUILD_BUNDLED_PLUGINS` | OFF | Build repository-maintained plugins |
| `AI_CORE_PLUGIN_VISION` | ON | Build OpenCV pre/post-process plugins |
| `WITH_ORT_ENGINE` | OFF | ONNX Runtime plugin |
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
// Normalization is (v - mean) / std — std_vals is a DIVISOR (a standard
// deviation), not a multiplier. Scaling 8-bit pixels to [0,1] wants 255,
// not 1/255.
arg.mean_vals = {0.f, 0.f, 0.f};
arg.std_vals = {255.f, 255.f, 255.f};
// Channel order the model was trained on. The preprocessor converts
// ImageView::format to this, so callers do not cvtColor themselves.
// Defaults to BGR888; ultralytics-style models are RGB.
arg.model_input_format = ImagePixelFormat::RGB888;
arg.hwc2chw = true;
arg.data_type = DataType::FLOAT32;
preproc_params.setParams(arg);

AlgoPostprocParams postproc_params;
AnchorDetParams det_arg;
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

- [docs/Framework.md](docs/Framework.md) — framework structure and design
- [docs/API.md](docs/API.md) — public API reference
- [docs/PluginGuide.md](docs/PluginGuide.md) — writing plugins, tensor contracts

## License

MIT
