# 插件开发指南与张量契约

AI Core 的三段流水线（预处理 / 推理 / 后处理）都是插件。新增算法 = 新 .cpp +
一行注册宏，无需改框架。本文说明怎么写插件，以及每个内置后处理插件的张量契约
（输入张量的名字数量 / shape / dtype）。

## 1. 动态插件入口

仓库内置插件与仓库外插件使用完全相同的动态库入口。核心不会自动注册任何
OpenCV、推理框架或硬件后端：

```cpp
#include <ai_core/plugin_manager.hpp>

ai_core::dnn::PluginManager::instance().load(
    "/path/to/libmy_amd_infer.so");
```

插件动态库导出 `ai_core_register_plugin_v1`，在其中向传入的 `PluginRegistry`
注册实现并填写 `PluginInfo`（API 版本、名称、版本、provider、能力列表）。入口
调用是事务性的：返回失败、抛出异常或 API 版本不匹配时，期间新增的注册项会
全部回滚，动态库也会立即卸载。插件载入成功后驻留到进程结束，不支持卸载，以保证
creator 和已经创建的对象不会指向被卸载的代码。

除逐个加载外，也可以扫描目录。自动发现依次扫描显式路径、
`AI_CORE_PLUGIN_PATH`（Linux/macOS 用 `:`，Windows 用 `;` 分隔）以及安装目录
`<libdir>/ai_core/plugins`。可发现的动态库名须包含 `ai_core_plugin_`：

```cpp
auto &plugins = ai_core::dnn::PluginManager::instance();
plugins.loadDirectory("/opt/my-product/plugins");
plugins.discover();
```

外部项目使用随包安装的 CMake helper：

```cmake
find_package(ai_core REQUIRED)
ai_core_add_plugin(my_amd_infer
    TYPE infer
    SOURCES amd_infer.cpp
    DEPENDENCIES amd_runtime::amd_runtime)
```

## 2. 注册一个源码内插件

```cpp
#include "ai_core/plugin_registrar.hpp"

REGISTER_PREPROCESS_ALGO(MyPreproc);   // IPreprocessPlugin 子类
REGISTER_INFER_ENGINE(MyEngine);       // IInferEnginePlugin 子类
REGISTER_POSTPROCESS_ALGO(MyPostproc); // IPostprocessPlugin 子类
```

宏把类名字符串与构造函数绑定进进程级 `PluginRegistry`。动态插件应在统一
入口函数中直接使用传入的 registry；宏主要保留给单体程序里的源码插件。

这些宏适合直接链接并由调用方显式执行注册代码的场景；可独立部署的插件应优先
使用上面的统一动态入口。

## 3. 插件接口契约

三类插件统一用 `InferErrorCode` 返回；异常只允许存在于插件内部，不得穿透
facade。`process` / `batchProcess` 是 `const` 且必须可重入——对象上不留可变的
per-call 状态，所有 scratch 走入参的 `TensorData` / `RuntimeContext`，这样一个
实例能并发服务多次调用（见 `docs/Framework.md` 线程模型）。

预处理把坐标变换信息写进 `RuntimeContext::frame_transform`
（`FrameTransformContext`：原始尺寸、ROI、缩放、padding），后处理读出来做坐标
还原。**不要自己从这些字段推缩放比**——用 `FrameTransformContext` 自带的成员，
它们是全仓唯一实现：

```cpp
ctx.sourceShape()              // 模型实际看到的区域（有 ROI 用 ROI，否则整帧）
ctx.scaleRatio()               // {x, y} 缩放比，等比模式下两轴相同
ctx.mapToSource({x, y})        // 模型输入坐标 -> 原图坐标（减 padding、加 ROI 偏移）
ctx.mapSizeToSource(w, h)      // 尺寸映射：只除缩放比，不减 padding 不加偏移
```

自由扩展数据放 `RuntimeContext::extras`（`DataPacket`）。

### 自定义结果类型

`AlgoOutput` 的 variant 尾部有一个 `DataPacket`，这是库外插件吐**自有结果类型**
的正门——不用改 ai-core 头文件，也不用借道 `RawModelOutput`/`TensorData`：

```cpp
// 插件侧
PoseRet pose{/* ... */};
DataPacket packet;
packet.setParam("pose", pose);
algo_output.setParams(std::move(packet));

// 消费侧
const auto *packet = output.getParams<DataPacket>();
const PoseRet pose = packet->getParam<PoseRet>("pose");
```

## 3. 后处理张量契约

后处理从 `TensorData`（张量名 → buffer + shape）读输入。`outputNames` 的**顺序**
即下表列出的顺序。dtype 由模型决定，下表是内置插件当前假设的类型。

| 插件 | 参数类型 | 输入张量（按 outputNames 顺序） | shape | dtype |
|---|---|---|---|---|
| `Yolov11Det` | `AnchorDetParams` | `[0]` 预测 | `[1, 4+nc, anchors]`（属性优先，内部转置） | FP32 或 FP16 |
| `NanoDet` | `AnchorDetParams` | `[0]` 预测 | `[1, anchors, nc+4]`（锚点优先：`scores..., x1,y1,x2,y2`） | FP32 |
| `RTMDet` | `AnchorDetParams` | `[0]` 框 / `[1]` 类别 | 框 `[1, anchors, 4]`（角点 x1,y1,x2,y2）；类别 `[1, anchors, nc]` | FP32 |
| `SoftmaxCls` | `GenericPostParams` | `[0]` **logits** | `[1, nc]` 或批量 `[N, nc]` | FP32 |
| `ArgmaxCls` | `GenericPostParams` | `[0]` **已归一化**分数 | `[1, nc]` 或批量 `[N, nc]` | FP32 |
| `FprCls` | `GenericPostParams` | `[0]` 分数 / `[1]` birads | `[1, nc]` / `[1, nb]`（批量首维 N） | FP32 |
| `OCRReco` | `GenericPostParams` | `[0]` 长度 / `[1]` argmax | 长度 `[N]`；argmax `[N, seq]`（CTC 折叠） | INT64 |
| `SemanticSeg` | `ConfidenceFilterParams` | `[0]` 类别图 | `[1, nc, h, w]`（批量 `[N, nc, h, w]`） | FP32 |
| `UNetDualOutputSeg` | `GenericPostParams` | `[0]` prob / `[1]` mask | 各 `[1, w, h]`（decoder 读 `shape[2]=h`, `shape[1]=w`） | FP32 |
| `RawModelOutput` | `GenericPostParams` | 全部原样透传 | 任意 | 任意 |

- `nc` = 类别数，`nb` = birads 数，`anchors` = 锚点数，`seq` = 序列长。
- `AnchorDetParams` 需 `condThre` + `nmsThre`；`ConfidenceFilterParams` 需
  `condThre`；`GenericPostParams` 只需 `outputNames`，另有可选的 `keepClassProbs`
  （分类插件是否在 `ClsRet::probs` 里保留完整分布，默认 false）。
- **`SoftmaxCls` 期望 logits，它自己会做一次 softmax。** 若模型已把 softmax 烘进
  计算图（ultralytics 的 `*-cls` 导出就是如此，输出和为 1），用它会 softmax 两次：
  softmax 单调所以**类别仍然正确**，坏掉的是**置信度**——会塌到接近 `1/nc` 的地板值。
  这类模型请用 `ArgmaxCls`，配置里换一个字符串即可，调用侧零改动。
- 检测类输出的坐标还原依赖预处理写入的 `FrameTransformContext`，缺失即返回
  `InferInvalidInput`。

## 4. 预处理契约

内置帧预处理插件（`CpuGenericPreprocess` / `CudaGenericPreprocess` /
`FrameWithMaskPreprocess`）消费 `FramePreprocessArg`，产出单个模型输入张量，名字
取 `inputNames[0]`，shape 依 `hwc2chw` 为 `{N,C,H,W}` 或 `{N,H,W,C}`。

两条容易踩的语义：

- **归一化是 `(v - mean_vals[c]) / std_vals[c]`，`std_vals` 是除数**（标准差），
  不是乘数。把 8bit 像素映射到 `[0,1]` 要填 `{255,255,255}` 而不是 `{1/255,...}`。
  写反不会报错，只会把像素放大到 0~65025，让 head 里的 sigmoid 全部饱和——症状
  看起来像后处理坏了。绑定参数时若发现 `std_vals` 全部 < 1 会打一条 warning。
- **`model_input_format` 会被真正消费。** 预处理按
  `ImageView::format → model_input_format` 做转换，调用方不必（也不应该）再自己
  `cvtColor`。默认是 `BGR888`（OpenCV 惯例），ultralytics 系模型填 `RGB888`。
  同通道数的互换（BGR↔RGB）折叠进归一化那一趟，零额外开销；通道数变化
  （GRAY↔BGR、BGRA→BGR）走一次 `cvtColor`。
`FrameWithMaskPreprocess` 把 mask 光栅化为额外通道，**调用方必须把
`inputShape.c` 设为含 mask 的真实通道数**（3 图 + 1 mask = 4）。

## 5. 新增一个检测头的最小步骤

1. 新建 `plugins/postproc/my-det/my_det.{hpp,cpp}`，继承 `FramePostprocBase<AnchorDetParams,
   true>`（`true` = 需要预处理变换上下文做坐标还原）。
2. 实现 `processTyped` / `batchProcessTyped` 两个纯虚 hook。
3. 在 .cpp 里 `REGISTER_POSTPROCESS_ALGO(MyDet);`。
4. JSON 配置里 `types.postproc` 填 `"MyDet"`，`postprocParams` 给
   `condThre/nmsThre/outputNames`。

不需要改任何工厂或分发代码，**也不需要改配置加载器**：`ai_core::config` 对未知的
模块名不再报错，它按 `postprocParams` 里出现的键推断参数族——

| 出现的键 | 推断出的参数族 |
|---|---|
| `condThre` + `nmsThre` | `AnchorDetParams` |
| 仅 `condThre` | `ConfidenceFilterParams` |
| 都没有 | `GenericPostParams` |

推断不合意时用 `"paramFamily": "anchorDet" / "confidenceFilter" / "generic"`
直接指定。预处理侧同理：`types.preproc` 填自定义插件名也能通过，因为
`FramePreprocessArg` 是唯一的预处理参数族。
