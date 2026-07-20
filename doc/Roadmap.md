# AI Core 重构路线图（v1.3 → v2.0）

本项目是个人长期打磨的推理框架，没有外部用户、没有兼容包袱。目标只有一个：达到顶尖个人项目水平，在需要起手一个产品时，它就是现成的顶级框架。

## 编排原则

没有兼容负担时，重构顺序应该反过来：**API break 最便宜的时刻就是现在**。所以先把公共 API 一次性打磨到终态（由内向外：数据类型 → 插件接口 → facade），再在稳定的 API 上投入测试、性能和并发——所有后续投入都落在最终形态上，一行不白写。

1. **先清算、后建设**：改名、语义修复、架构合一放最前面，越晚做越贵。
2. **重投入跟在 API 终态之后**：全量单测、benchmark 基线在 v1.4 之后铺开；v1.3 只带轻安全网（CI + sanitizer + 现有集成测试保持绿色）。
3. **每个版本有验收标准**，性能类工作必须 benchmark 前后对比。
4. 最终里程碑 v2.0 的定义是："从这个框架起手一个新产品，一天内跑通全链路。"

---

## v1.3 — 清算与合一

> 主题：所有已知的正确性缺陷、命名硬伤、架构冗余一次清掉。这一版结束后，框架里不再有"知道不对但先留着"的东西。

### 正确性修复

- [x] **预处理/后处理异常泄漏**：`AlgoPreproc::Impl::process` / `batchProcess` 及 postproc 对应方法没有 try/catch，插件内部 `throw` 会穿透 `AlgoInference::infer`。facade 边界统一 catch-all 转错误码。
- [x] **ORT 后端 `infer` / `terminate` 竞态**：`infer` 不持锁、`terminate` 持锁 reset session，并发即 use-after-free。统一锁策略。
- [x] **初始化检查失效**：`AlgoInference::Impl::infer` 判空的三个成员指针在构造函数就创建了，永远非空。加 `m_initialized`，未初始化 fail fast。

### API 清算（一次性 break）

- [x] 命名：`IPreprocssPlugin` → `IPreprocessPlugin`、`IPostprocssPlugin` → `IPostprocessPlugin`、`DaulRawSegRet` → `DualRawSegRet`、`accelerator_buffer_impl.hpp` 更名（它是接口不是 impl）。
- [x] 可移植性：`u_char` → `uint8_t`（公共头在 MSVC 下编译不过）；移除 deprecated 的 `std::codecvt`。
- [x] `BBox` 改值语义：去掉 `shared_ptr<cv::Rect>`（每框一次堆分配 + 原子引用计数，`DetRet` 拷贝变浅共享）。
- [x] **插件接口统一错误码**：`IPreprocessPlugin` / `IPostprocessPlugin` 的 `bool` 返回改为 `InferErrorCode`，异常只存在于插件内部。

### 架构合一

- [x] **分发机制二合一**：删除"字符串工厂 → enum switch"双层分发。`Yolov11Det` / `RTMDet` / `NanoDet` / `SoftmaxCls` / 各 preprocessor 直接注册为工厂插件，`AnchorDetPostproc` / `CVGenericPostproc` / `FramePreprocess` 这些 enum 分发壳删除。新增算法 = 新文件 + 一行注册宏。
- [x] **RuntimeContext 类型化**：preproc → postproc 的 `FrameTransformContext` 传递改为类型化槽位，消灭 `"preproc_runtime_args"` 魔法字符串；`DataPacket` 仅保留给自由扩展。
- [x] **注册机制健壮化**：提供显式 `registerDefaultPlugins()` 入口，静态/动态链接皆可用，不再依赖"恰好是 SHARED 库"才成立的静态初始化。

### 轻安全网与工程清理

- [x] CI（GitHub Actions）：GCC + Clang，Debug 跑 ASan/UBSan，clang-format 检查；现有 5 个集成测试保持绿色。（workflow 已就绪，ASan/UBSan 已在本地 ORT 路径验证全绿；首次 push 后如有环境差异需微调）
- [x] `escaleResizeWithPad` 坐标还原单测（检测框正确性的命门，且此逻辑不受 API 重塑影响，先锁住）。
- [x] `ai_core-config.cmake.in` 加 `find_dependency`；`CMAKE_CUDA_ARCHITECTURES` 去硬编码；`install/` 移出仓库（确认从未被 git 跟踪且已在 .gitignore，无需改动）。

### 验收标准

- 代码里搜不到 `u_char`、拼写错误类名、enum 分发 switch。
- 任何插件 throw 不穿透 facade；sanitizer 全绿。
- 新增一个检测头只需要：一个新 .cpp + 注册宏。

---

## v1.4 — 数据层重塑

> 主题：公共 API 的最后一块深水区。这一版结束后，公共 API 达到终态，此后只加不改。

### 任务

- [x] **`ImageView` 抽象**：公共 API 以自有 `ImageView`（data ptr + shape + stride + format）替代 `cv::Mat` / `cv::Rect`，OpenCV 降为实现细节；`ai_core/opencv_interop.hpp` 提供零成本互转。这是支持非视觉模态（音频、文本、任意张量输入）的前提——"AI Core" 这个名字才名副其实。
- [x] **`TensorData` v2**：`datas` 与 `shapes` 两个平行 map 易失同步，聚合为单一 `Tensor` 类型（buffer + shape + dtype）；容器换按插入序 small-vector + 名字查找（输入输出通常 1~3 个，`std::map` 纯浪费）。
- [x] **`TypedBuffer` 收口**：工厂方法梳理，`resize` 的分歧语义（Pageable 保数据 / Pinned、GPU 破坏性）在类型层面显式化。
- [x] **配置与数据分离**：不变的 preproc/postproc 参数在 `initialize` 时绑定并校验一次，`infer()` 只带数据（保留可选 per-call override）。
- [x] **日志头瘦身**：`logger.hpp` 拆为轻接口 + 实现，公共头不再连带 `<iostream>` `<fstream>` `<thread>`。

### 验收标准

- 公共头 `api/ai_core/` 零第三方类型、零 POSIX 类型，MSVC 可编译（哪怕暂不官方支持 Windows）。
- 用纯指针数据（不经 OpenCV）能走通完整推理链路。
- 声明此版为 **API 终态**：写进 CHANGELOG，此后接口只加不改。

---

## v1.5 — 测试体系与基线

> 主题：在终态 API 上把安全网织满。这是后续性能与并发工作的前提，测试写一遍就是最终版。

### 任务

- [ ] **单元测试全覆盖**：`TypedBuffer`（拷贝/移动/引用语义、resize、类型校验）、`Tensor`/`TensorData`、`DataPacket`、`ParamCenter`、`Factory`、`ImageView` 互转、`convertLayout`、各 postproc 解码逻辑（用构造的张量数据，不依赖模型资产）。核心组件行覆盖 ≥ 80%，CI 强制。
- [ ] **集成测试矩阵**：det / cls / seg / OCR × ORT / TRT / NCNN，模型资产脚本化下载。
- [ ] **benchmark 基线化**：单帧预处理、端到端、各后端吞吐，结果存档进仓库，每版对比，性能回退 CI 报警。
- [ ] **线程安全审计**：每个公共类标明"可并发 / 需外部同步 / 单线程"，写进头文件注释，为 v1.7 立契约。

### 验收标准

- 不下载任何模型也能跑完全部单测。
- 覆盖率与 benchmark 对比在 CI 里可见。

---

## v1.6 — 性能：数据通路瘦身

> 主题：benchmark 驱动，消灭热路径上的多余拷贝与分配。

### 任务

- [ ] **CPU 预处理单趟化**：现状一帧至少 4 次全图遍历/分配（`clone` → `convertTo` → `split`/`merge` 归一化 → `convertLayout`）。归一化合并进 `convertTo(alpha, beta)`，直接写目标 buffer。
- [ ] **FP16 路径去双拷贝**：去掉中间 FP32 vector 与末端 `vector<uint8_t>` 拷贝，直接落 `TypedBuffer`。
- [ ] **ORT 输出零拷贝**：IOBinding + 预分配输出缓冲（`max_output_buffer_sizes` 字段已存在但 ORT 未用）。
- [ ] **热路径堆分配清零**：preprocessor 实例、后处理分发对象移到 initialize 阶段持有；`RuntimeContext` 复用。
- [ ] **ORT 线程配置可配**：`SetIntraOpNumThreads(hardware_concurrency())` 硬编码在多实例场景互相打架。

### 验收标准

- YOLO 640×640 基准，单帧 CPU 预处理耗时较 v1.5 基线下降 ≥ 40%。
- 端到端路径的每次"分配 + 拷贝"可枚举且有存在理由。

---

## v1.7 — 并发与异步落地

> 主题：让已建好的异步基础设施（`IAsyncInferEngine` / `IExecutionContext` / TRT stream，500+ 行且有测试）从孤岛变成正门可达。

### 任务

- [ ] **facade 暴露异步**：`AlgoInference::getAsyncEngine()`（后端不支持返回空），不再需要绕过 facade 玩 `dynamic_pointer_cast`。
- [ ] **TRT 同步路径去大锁**：`TrtAlgoInference::infer` 的全局 mutex 换成 execution context pool，多线程并发各拿各的 context。
- [ ] **ORT 并发审计**：`Session::Run` 本身线程安全，去掉多余串行化点。
- [ ] **多线程吞吐 benchmark**：1/2/4/8 线程压测入基线；TSan 压测全绿。
- [ ] **异步 example**：context pool + pinned buffer + CUDA graph 的完整流水线示例——这是框架的性能卖点，必须有可运行的展示。

### 验收标准

- GPU 不饱和前提下，TRT 4 线程吞吐 ≥ 单线程 3 倍。
- 异步路径从公共 API 可达，有文档有示例。

---

## v2.0 — 产品起手能力

> 主题：里程碑定义——"从这个框架起手一个新产品，一天内跑通全链路。" 这一版做的不是框架本身，而是围绕框架的起手体验。

### 任务

- [ ] **配置模块**：`examples/algo_config_parser` 升级为正式可选模块 `ai_core::config`，全部算法参数支持 JSON 定义，加载即用——新产品的算法编排不写 C++。
  - 已知问题（v1.3 发现）：parser 与 `assets/conf/*.json` 键风格不一致（parser 读 `preproc_params`/`input_names` 等 snake_case，JSON 写 `preprocParams`/`inputNames` 等 camelCase），preproc/postproc 参数从未被解析，OCR 示例因此跑不通。升级为正式模块时统一键风格并加 schema 校验。
- [ ] **错误详情通道**：`InferErrorCode::to_string`；facade 提供错误上下文（哪个张量、期望什么、拿到什么），排查不靠翻日志。
- [ ] **示例即模板**：det / cls / seg / OCR / 异步流水线示例整理成可直接复制起手的 starter 结构（`examples/starter/`），带自己的 CMakeLists，验证 `find_package(ai_core)` 消费路径。
- [ ] **文档体系**：Doxygen 站点、架构文档更新（Framework.md 对齐终态）、插件开发指南、各 postproc 张量契约（名字/shape/dtype）、线程模型文档。
- [ ] **CMake 组件化**：后端拆为可选组件（`find_package(ai_core COMPONENTS ort trt)`）；CMake Presets；显式源文件列表替代 `GLOB_RECURSE`。
- [ ] **一键环境**：脚本化拉起依赖 + 构建 + 测试（新机器从 clone 到全绿一条命令）。

### 可选延伸（backlog，按需要再排）

- Python bindings（快速原型验证时有价值，但增加维护面，等真实需求出现再做）。
- 更多模态的内置插件（音频前处理、文本 tokenize），在 ImageView/Tensor 抽象上按需生长。

### 验收标准

- 新机器：clone → 一条命令 → 全部测试与示例通过。
- 起手实测：复制 starter、换一个新模型、改 JSON 配置，一天内跑通新产品原型。

---

## 版本总览

| 版本 | 主题 | 相对工作量 | 关键产出 |
|---|---|---|---|
| v1.3 | 清算与合一 | ★★★ | 零已知缺陷、单一分发机制、CI 安全网 |
| v1.4 | 数据层重塑 | ★★★ | **公共 API 终态**（ImageView、Tensor） |
| v1.5 | 测试体系与基线 | ★★☆ | 覆盖 ≥80%、benchmark 基线、线程契约 |
| v1.6 | 数据通路性能 | ★★☆ | 预处理 -40%、零多余拷贝 |
| v1.7 | 并发与异步 | ★★★ | 多线程近线性、异步正门可达 |
| v2.0 | 产品起手能力 | ★★☆ | 一天起手一个产品原型 |

依赖关系：v1.5 的测试投入必须等 v1.4 的 API 终态（否则测试写两遍）；v1.6/v1.7 的性能与并发必须踩在 v1.5 的基线与安全网上。v1.3 与 v1.4 之间顺序可微调，但都必须在 v1.5 之前完成。
