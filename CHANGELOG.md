# Changelog

本项目无外部用户、无兼容包袱；v1.4 之前的接口变更一律不留兼容别名。**v1.4 起公共 API 终态，此后只加不改**——v2.1 是唯一的例外，见下。

## v2.1 — 下游反馈收口（2026-08）

来自 ai-sdk 集成的 16 条反馈。**本版含 break**，是 v1.4「只加不改」承诺的一次有意破例：两条都属于「字段名/字段语义会导致静默错误」，留着比改掉代价更大。

**break（无兼容别名）：**

- **`FramePreprocessArg::norm_vals` → `std_vals`**。语义一直是 `(v - mean) / norm`，即**除数**；但 NCNN/MNN 里同名字段是**乘数**，直觉会写成 `1/255`，结果像素被放大到 0~65025、head 里 sigmoid 全部饱和，且**不报任何错**。改名把语义写进名字，头文件里补上公式，另加一条运行期告警：8bit 输入且 `std_vals` 全部 < 1 时提示疑似写反。
- **预处理开始真正消费 `ImageView::format`**。此前 CPU 路径完全不做通道序转换（`format` 是装饰性字段），ncnn 路径却**无条件** `BGR2RGB`——同一份配置换个预处理插件结果不同，且 BGR 喂给 RGB 训练的模型是静默掉点。新增 `FramePreprocessArg::model_input_format`（默认 `BGR888`），预处理按 `image.format → model_input_format` 转换。同通道数互换折叠进归一化那一趟（零开销），通道数变化走 `cvtColor`。调用方应删掉自己的 `cvtColor`，改为如实声明格式。JSON 键 `preprocParams.inputFormat`。
- **包配置安装路径** `<prefix>/share/` → `<prefix>/lib/cmake/ai_core/`，与 ai-pipe / ai-stream 一致；消费者不再需要手动把 `share/` 加进 `CMAKE_PREFIX_PATH`。同时把 `configure_file` 换成 `configure_package_config_file`——此前 `@PACKAGE_INIT@` 展开成空，生成的 config 不可重定位。

**新增（只加不改）：**

- **`AlgoOutput` variant 尾部加 `DataPacket`**：库外插件吐自有结果类型的正门（`packet.setParam("pose", PoseRet{...})`），不必改 ai-core 头文件、不必重编下游、不必借道 `RawModelOutput`/`TensorData`。
- **`ArgmaxCls` 内置后处理**：给已归一化输出取 top-1、分数原样透传。ultralytics 的 `*-cls` 导出把 softmax 烘进了计算图，用 `SoftmaxCls` 会 softmax 两次——类别对、**置信度塌到 `1/nc` 地板值**，按置信度做阈值或表决的下游会中招。`SoftmaxCls` 的文档补上「期望 logits」这个前提。
- **`ClsRet::probs` + `GenericPostParams::keep_class_probs`**：完整类别分布本来就在后处理里算出来了，开关打开就不丢（默认关，行为与之前完全一致）。滑窗表决类下游不必再对硬判决表决。
- **`FrameTransformContext` 收口坐标映射**：`sourceShape()` / `scaleRatio()` / `mapToSource()` / `mapSizeToSource()`。此前这套推导在 4 个内置解码器里各抄了一遍，下游还得再抄第五遍。`utils::scaleRatio` 已删除，全仓单一实现。
- **配置层去掉硬编码白名单**：未知的 postproc 模块名不再 `fail`，改按 `postprocParams` 出现的键推断参数族（`condThre`+`nmsThre` → AnchorDet；仅 `condThre` → ConfidenceFilter；否则 Generic），可用 `paramFamily` 显式覆盖；preproc 侧白名单直接移除。自定义插件从此可被配置驱动——内建算法与自定义算法走同一条配置路径。
- **`REGISTER_*` 宏命名空间无关**：宏体里所有名字改全限定，库外插件不再需要同时 `using namespace ai_core;` 和 `ai_core::dnn;`。
- 安装导出的 config 目标别名统一为 `ai_core::config`（此前 build tree 叫 `ai_core::config`、安装后叫 `ai_core::ai_core_config`）。

**修复：**

- `scripts/bootstrap.sh` / README / `scripts/x86_build.sh` 里三处指向三个不同 release tag 的依赖包链接，其中两个已 404，照 README 走必然卡在 `tar: not in gzip format`。改为与 CI 同源：ONNX Runtime 取 Microsoft 官方发布版，OpenCV 用系统包。`curl` 加 `-f` 让 404 在下载阶段就失败。
- 顺带消除 OpenCV 双实例风险：不再解出 vendored OpenCV。`load_opencv()` 在检测到 vendored 与系统 OpenCV 并存时告警；bootstrap 结尾断言 `ldd | grep -c opencv_core == 1`。
- CI 里 patch onnxruntime cmake export 的两条 `sed`（`lib64/`→`lib/`、`include/onnxruntime`→`include`）下沉进 `scripts/bootstrap.sh`，手动流程与 CI 一致。
- `load_3rdparty.cmake` 的 `load_tensorrt()` 用 `CMAKE_SOURCE_DIR` 而非 `PROJECT_SOURCE_DIR`，super-build + TRT 组合下找不到 `FindTensorRT.cmake`。
- `UNetDualOutputSeg` 的 `DualRawSegRet::ratio` 此前按整帧宽度推导，忽略 ROI 与等比缩放；改用统一的 `scaleRatio()`。
- 4 处 doxygen `@file` 与真实文件名不符。
- `version.hpp` 停在 1.2.0 而 CHANGELOG 已到 v2.0，安装出来的包版本是错的。CI 新增一条断言，让两者不能再各走各的。

## v2.0 — 产品起手能力（2026-07）

新增（公共 API 只加不改）：

- **`ai_core::config` 模块**（`<ai_core/config/algo_config.hpp>`，`BUILD_AI_CORE_CONFIG`）：JSON 加载整条流水线 + schema 校验（`ConfigError`）。统一 camelCase 键，修复 v1.3 记录的 snake/camelCase 不一致 bug，OCR 示例端到端跑通。
- **`to_string(InferErrorCode)`** + `operator<<`；facade 错误日志带 stage + 错误码。
- **`AlgoInference::getAsyncEngine()` / `AlgoInferEngine::getAsyncEngine()`**（v1.7）：异步正门。
- `AlgoInferParams.intra/inter_op_num_threads`（v1.6）：ORT 线程可配。
- `examples/starter/`（find_package 消费路径）、`examples/async_pipeline/`（异步流水线）、`doc/PluginGuide.md`（张量契约）、`Doxyfile`、`CMakePresets.json`、`scripts/{bootstrap,fetch_models,coverage}.sh`。

## v1.7 — 并发与异步（2026-07）

- **TRT 同步 infer 去大锁**：`TrtAlgoInference::infer` 全局 mutex → execution context pool（每 context 独立 CUDA stream）。**break**：并发语义从「串行」变「并发」。
- 异步正门 `getAsyncEngine()`；ORT 并发审计（已最优）；多线程吞吐 benchmark；TSan 我方代码全绿（`tests/tsan.supp`）。

## v1.6 — 数据通路性能（2026-07）

- CPU 单帧预处理融合单趟化：3.16ms → 1.53ms（-51.6%）；FP16 去双拷贝。
- ORT 静态输出 IOBinding 零拷贝；`intra/inter_op_num_threads` 可配。
- **break**：`TypedBuffer::resize` → `resizeDiscard` / `resizePreserving`（已在 v1.4 落地）。

## v1.5 — 测试体系与基线（2026-07）

- 核心组件行覆盖 82%（`scripts/coverage.sh`，CI 门禁 80%）；集成矩阵 + 模型 provisioning 脚本化；benchmark 基线存档；线程契约进头文件（`@par Thread safety`）。无 API 变更。

## v1.4 — 数据层重塑（2026-07）

**本版本起公共 API 进入终态：此后接口只加不改。**

公共头 `api/ai_core/` 达成零第三方类型、零 POSIX 类型（唯一例外是 opt-in 的
`opencv_interop.hpp`，它是显式的 OpenCV 互转入口）。纯指针像素数据可以不经
OpenCV 走通完整推理链路（`AlgoInferenceTest.PurePointerPath` 锁定该能力）。

### Breaking Changes

- **`ImageView` 抽象**：`FrameInput` 由 `shared_ptr<cv::Mat>` + `shared_ptr<cv::Rect>`
  改为非拥有的 `ImageView`（data + width/height + stride + `ImagePixelFormat`）+
  `optional<Rect>`。几何类型改为自有 `Point` / `Point2f` / `Rect` 值类型；
  `Contour` 基于 `ai_core::Point`。互转走 `ai_core/opencv_interop.hpp`
  （`viewFromMat` / `matFromView` / `toCv` / `fromCv`，零拷贝）。
- **`TensorData` v2**：`datas` / `shapes` 平行 map 聚合为单一 `Tensor`
  （name + `TypedBuffer` + shape），按插入序存放于扁平 vector、按名字线性查找。
  接口：`set` / `find` / `at` / `contains` / 迭代器。
  `DualRawSegRet::mask/prob` 改为拥有数据的 `Tensor`（原实现包装推理输出缓冲的
  cv::Mat 头，TensorData 释放后即悬垂）。
- **`TypedBuffer` 收口**：`createFromGpu` 双重语义拆分为 `allocateGpu`（分配）与
  `wrapGpu`（非拥有包装）；`createFromCpuRef` 改为 `wrapCpu`（纯非拥有，删除
  `manage_memory` 机制）；删除 `setCpuData` / `setGpuDataReference`；`resize` 拆为
  `resizeDiscard`（全类型统一破坏性，输出缓冲专用）与 `resizePreserving`
  （仅 CPU pageable，其余抛 `std::logic_error`）。
- **配置与数据分离**：`AlgoPreproc` / `AlgoPostproc` / `AlgoInference` 的
  `initialize` 接收并绑定参数（做一次结构校验，拒绝 monostate）；
  `infer` / `process` 只带数据，保留可选的 per-call override 指针。
  `AlgoManager::infer(name, input, output)`。
- **日志头瘦身**：`logger.hpp` 只保留轻接口（Logger pimpl 化，热路径
  `isEnabled` 仍为内联原子读）；`<iostream>` `<fstream>` `<thread>` 等重头文件
  全部移入 `logger.cpp`。`LogEntry::thread_id` 改为 `uint64_t`。删除未使用的
  C++20 `source_location` 探测与 `LOG_*S` 旧别名宏（规范名为 `LOG_*_S`）。

### 迁移提示（对照旧代码）

| 旧 | 新 |
|---|---|
| `frame_input.image = std::make_shared<cv::Mat>(m)` | `frame_input.image = interop::viewFromMat(m)` |
| `frame_input.input_roi = std::make_shared<cv::Rect>(...)` | `frame_input.roi = Rect{...}` |
| `tensor_data.datas.at(n)` / `.shapes.at(n)` | `tensor_data.at(n).buffer` / `.at(n).shape` |
| `TypedBuffer::createFromGpu(t, size)` | `TypedBuffer::allocateGpu(t, size)` |
| `buf.resize(n)` | `buf.resizeDiscard(n)` 或 `buf.resizePreserving(n)` |
| `infer(input, pre, post, out)` | `initialize(pre, post)` + `infer(input, out)` |

## v1.3 — 清算与合一

正确性修复（异常泄漏、ORT 竞态、初始化检查）、API 清算（命名、可移植性、
`BBox` 值语义、插件错误码统一）、架构合一（单一工厂分发、`RuntimeContext`
类型化、显式插件注册）、CI + sanitizer 安全网。详见 `doc/Roadmap.md`。
