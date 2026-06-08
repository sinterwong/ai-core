# AI Core

<p align="center">
  <img src="assets/icon/logo.jpeg" alt="ai-core Logo" width="500"><br/>
</p>

<p align="center">C++ AI 推理框架</p>

[English](README_EN.md) | [简体中文](README.md)

![Version](https://img.shields.io/badge/version-1.2.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)

AI Core 是一个用于在多种推理后端（ONNX Runtime、NCNN、TensorRT）上部署模型的 C++ 库。流水线由三段组成：预处理、推理、后处理。每一段都通过插件方式注册，按名称拼装，便于扩展。

## 数据流

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

`TensorData` 是插件之间传递的张量集合。`AlgoInput` / `AlgoOutput` 是流水线两端的输入/输出。

## 编译

### 依赖

- C++20 兼容的编译器（GCC 11+、Clang 14+、MSVC 19.30+）
- CMake 3.18+
- OpenCV 4.x
- ONNX Runtime（默认启用）
- 可选：NCNN、TensorRT、CUDA Toolkit

### 拉取与构建

```bash
git clone --recurse-submodules https://github.com/sinterwong/ai-core.git
cd ai-core
mkdir -p 3rdparty/target/

# 下载预编译的第三方依赖
curl -L https://github.com/sinterwong/ai-core/releases/download/v1.1.1-alpha/dependency-Linux_x86_64.tgz -o dependency.tgz
tar -xzf dependency.tgz -C 3rdparty/target/

mkdir build && cd build
cmake .. -DBUILD_AI_CORE_EXAMPLES=ON -DBUILD_AI_CORE_TESTS=ON \
         -DWITH_ORT_ENGINE=ON -DWITH_NCNN_ENGINE=ON -DWITH_TRT_ENGINE=OFF

cmake --build . -j
cmake --install .
```

CMake 选项：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `BUILD_AI_CORE_TESTS` | OFF | 构建单元测试 |
| `BUILD_AI_CORE_BENCHMARKS` | OFF | 构建 benchmark |
| `BUILD_AI_CORE_EXAMPLES` | OFF | 构建示例 |
| `WITH_ORT_ENGINE` | ON | ONNX Runtime 后端 |
| `WITH_NCNN_ENGINE` | OFF | NCNN 后端 |
| `WITH_TRT_ENGINE` | OFF | TensorRT 后端 |

## 使用

`AlgoInference` 是流水线入口，构造时给出三段插件的名字和推理参数：

```cpp
#include "ai_core/algo_inference.hpp"
#include "ai_core/algo_types.hpp"

using namespace ai_core;

AlgoModuleTypes modules{
    "FramePreprocess",     // 预处理插件
    "OrtAlgoInference",    // 推理后端
    "AnchorDetPostproc"    // 后处理插件
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
    // 处理错误
}

if (auto* det = output.getParams<DetRet>()) {
    for (const auto& box : det->bboxes) {
        // ...
    }
}

algo.terminate();
```

更完整的例子参考 `examples/generic_image_infer.cpp`、`examples/ocr/` 和 `tests/`。

## 文档

- [doc/Framework.md](doc/Framework.md) — 框架结构与设计
- [doc/API.md](doc/API.md) — 公共 API

## 许可

MIT
