# AI Core

<p align="center">
  <img src="assets/icon/logo.jpeg" alt="ai-core Logo" width="500"><br/>
</p>

<p align="center">C++ AI 推理框架</p>

[English](README_EN.md) | [简体中文](README.md)

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)

AI Core 用插件组合预处理、推理和后处理流水线，支持 ONNX Runtime、NCNN、
TensorRT 以及库外自定义后端。

## v2 依赖边界

默认构建只有 `libai_core`。它不发现、编译或链接任何第三方库，动态依赖仅限
C++ 与操作系统基础运行库。config、每个官方插件以及开发工具都是独立的 opt-in
组件；打开哪个组件，才解析哪个组件拥有的依赖。

- `third_party/config/`：config 源码依赖。
- `third_party/plugins/`：官方插件源码依赖。
- `third_party/testing/`、`third_party/benchmarking/`：仅开发工具使用。
- `.deps/<OS>_<ARCH>/`：不入 Git 的预编译 SDK，例如 ONNX Runtime、NCNN、
  TensorRT。

所有 Git submodule 固定到 gitlink commit，不跟踪浮动分支。无需再对仓库执行
递归 submodule 初始化；使用 `scripts/deps.sh init <profile>` 按需拉取。

## 快速开始

只构建零第三方依赖的 core：

```bash
git clone https://github.com/sinterwong/ai-core.git
cd ai-core
cmake --preset core
cmake --build --preset core
cmake --install build/core
```

构建某个组件时先初始化同名依赖 profile：

```bash
# JSON config
scripts/deps.sh init config
cmake --preset config
cmake --build --preset config

# 仓库固定版本的 OpenCV 插件
scripts/deps.sh init vision
cmake --preset vision
cmake --build --preset vision

# 常用开发组合：config + vision + ONNX Runtime + tests
# developer preset 使用系统 OpenCV。
scripts/deps.sh init config onnxruntime testing
cmake --preset developer
cmake --build --preset developer
ctest --preset developer
```

`scripts/bootstrap.sh` 提供完整的一键开发构建，默认使用固定的 bundled OpenCV；
可用 `--opencv-provider SYSTEM` 切换到系统 OpenCV。

```bash
scripts/bootstrap.sh
```

可用 profile：`core`、`config`、`vision`、`onnxruntime`、`ncnn`、`tensorrt`、
`decryption`、`testing`、`benchmarking`、`developer`。NCNN/TensorRT 不提供通用的
公开下载包，可分别通过 `AI_CORE_NCNN_ARCHIVE`、`AI_CORE_TENSORRT_ARCHIVE`
向 profile 提供 SDK tar 包，或直接设置对应的 CMake root。

## CMake 选项

所有选项默认 `OFF`。

| 选项 | 说明 |
| --- | --- |
| `AI_CORE_BUILD_CONFIG` | 构建 JSON config 模块 |
| `AI_CORE_BUILD_PLUGIN_PREPROC_OPENCV` | 构建 OpenCV 预处理插件 |
| `AI_CORE_BUILD_PLUGIN_POSTPROC_OPENCV` | 构建 OpenCV 后处理插件 |
| `AI_CORE_BUILD_PLUGIN_ONNXRUNTIME` | 构建 ONNX Runtime 推理插件 |
| `AI_CORE_BUILD_PLUGIN_NCNN` | 构建 NCNN 推理插件 |
| `AI_CORE_BUILD_PLUGIN_TENSORRT` | 构建 TensorRT 推理插件 |
| `AI_CORE_ENABLE_MODEL_DECRYPTION` | 为已启用的推理插件加入模型解密支持 |
| `AI_CORE_BUILD_TESTS` | 构建按依赖边界拆分的测试 |
| `AI_CORE_BUILD_BENCHMARKS` | 构建 benchmark |
| `AI_CORE_BUILD_EXAMPLES` | 构建示例（要求 config） |

依赖定位参数：

- `AI_CORE_OPENCV_PROVIDER=BUNDLED|SYSTEM`，默认 `BUNDLED`。
- `AI_CORE_OPENCV_ROOT`，为 `SYSTEM` provider 指定可重定位的 OpenCV SDK 根目录。
- `AI_CORE_DEPS_ROOT`，默认 `.deps/<OS>_<ARCH>`。
- `AI_CORE_ONNXRUNTIME_ROOT`、`AI_CORE_NCNN_ROOT`、`AI_CORE_TENSORRT_ROOT`。
- `AI_CORE_CUDA_ARCHITECTURES`。

缺失依赖会在 configure 阶段报出精确路径、对应 profile 和可覆盖的 root，不会扫描
隐式系统目录或回退到另一套 SDK。

### Android arm64 交叉编译

Android 预编译 SDK 建议使用以下布局：

```text
.deps/Android_aarch64/
├── ncnn/
├── onnxruntime/
└── opencv/              # OpenCV Android SDK 根，内部包含 sdk/native/jni
```

设置 `ANDROID_NDK_HOME` 后，仓库内的 `.vscode/settings.json` 可直接由 VS Code
CMake Tools 配置和构建 arm64-v8a。等价的关键参数是
`AI_CORE_DEPS_ROOT=.deps/Android_aarch64`、
`AI_CORE_OPENCV_PROVIDER=SYSTEM` 和
`AI_CORE_OPENCV_ROOT=.deps/Android_aarch64/opencv`。显式 SDK package 在交叉编译时
不会被重映射到 NDK sysroot。

## 测试

```bash
scripts/deps.sh init testing
cmake --preset core-tests
cmake --build --preset core-tests
ctest --preset core-tests

# 按标签执行：core / config / vision / backend
ctest --test-dir build/developer -L backend --output-on-failure

# core 行覆盖率门禁
scripts/coverage.sh
```

只打开 tests 时只产生 core suite，不会引入 OpenCV 或推理 SDK。模型资产仅由
backend 集成测试使用；机器相关的 TensorRT engine 通过 `scripts/fetch_models.sh`
本地生成。

## 安装与消费

安装组件名为 `runtime`、`development`、`config`、`plugin-preproc-opencv`、
`plugin-postproc-opencv`、`plugin-onnxruntime`、`plugin-ncnn`、`plugin-tensorrt`、
`tests`、`benchmarks`、`examples` 和 `assets`。

```cmake
find_package(ai_core 2.0 REQUIRED COMPONENTS core config)
target_link_libraries(my_app PRIVATE ai_core::ai_core ai_core::config)
```

插件还会导出 `ai_core::plugin_preproc_opencv`、
`ai_core::plugin_postproc_opencv`、`ai_core::plugin_onnxruntime`、
`ai_core::plugin_ncnn`、`ai_core::plugin_tensorrt`。运行期插件目录由 package config
变量 `ai_core_PLUGIN_DIR` 给出。外部插件使用 `AICorePlugin.cmake` 中的
`ai_core_add_plugin()` 创建。

## 使用与文档

应用通过 `PluginManager` 显式加载所需插件，再用 `AlgoInference` 按注册名组合三段
流水线。公共 API 不暴露第三方类型；需要 OpenCV 互转时显式包含
`<ai_core/opencv_interop.hpp>`。

- [docs/Framework.md](docs/Framework.md) — 框架结构与线程模型
- [docs/API.md](docs/API.md) — 公共 API
- [docs/PluginGuide.md](docs/PluginGuide.md) — 插件开发与张量契约
- [examples/starter](examples/starter) — 已安装 package 的最小消费工程

## 许可

MIT
