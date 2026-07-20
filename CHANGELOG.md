# Changelog

本项目无外部用户、无兼容包袱；v1.4 之前的接口变更一律不留兼容别名。

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
