# 插件开发指南与张量契约

AI Core 的三段流水线（预处理 / 推理 / 后处理）都是插件。新增算法 = 新 .cpp +
一行注册宏，无需改框架。本文说明怎么写插件，以及每个内置后处理插件的张量契约
（输入张量的名字数量 / shape / dtype）。

## 1. 注册一个插件

```cpp
#include "ai_core/plugin_registrar.hpp"

REGISTER_PREPROCESS_ALGO(MyPreproc);   // IPreprocessPlugin 子类
REGISTER_INFER_ENGINE(MyEngine);       // IInferEnginePlugin 子类
REGISTER_POSTPROCESS_ALGO(MyPostproc); // IPostprocessPlugin 子类
```

宏把类名字符串与构造函数绑定进对应工厂。内置插件由
`registerDefaultPlugins()`（facade `initialize()` 自动调用）注册；自定义插件在
自己的 .cpp 里执行上述宏即可，静态/动态链接皆可用。

## 2. 插件接口契约

三类插件统一用 `InferErrorCode` 返回；异常只允许存在于插件内部，不得穿透
facade。`process` / `batchProcess` 是 `const` 且必须可重入——对象上不留可变的
per-call 状态，所有 scratch 走入参的 `TensorData` / `RuntimeContext`，这样一个
实例能并发服务多次调用（见 `doc/Framework.md` 线程模型）。

预处理把坐标变换信息写进 `RuntimeContext::frame_transform`
（`FrameTransformContext`：原始尺寸、ROI、缩放、padding），后处理读出来做坐标
还原。自由扩展数据放 `RuntimeContext::extras`（`DataPacket`）。

## 3. 后处理张量契约

后处理从 `TensorData`（张量名 → buffer + shape）读输入。`outputNames` 的**顺序**
即下表列出的顺序。dtype 由模型决定，下表是内置插件当前假设的类型。

| 插件 | 参数类型 | 输入张量（按 outputNames 顺序） | shape | dtype |
|---|---|---|---|---|
| `Yolov11Det` | `AnchorDetParams` | `[0]` 预测 | `[1, 4+nc, anchors]`（属性优先，内部转置） | FP32 或 FP16 |
| `NanoDet` | `AnchorDetParams` | `[0]` 预测 | `[1, anchors, nc+4]`（锚点优先：`scores..., x1,y1,x2,y2`） | FP32 |
| `RTMDet` | `AnchorDetParams` | `[0]` 框 / `[1]` 类别 | 框 `[1, anchors, 4]`（角点 x1,y1,x2,y2）；类别 `[1, anchors, nc]` | FP32 |
| `SoftmaxCls` | `GenericPostParams` | `[0]` logits | `[1, nc]` 或批量 `[N, nc]` | FP32 |
| `FprCls` | `GenericPostParams` | `[0]` 分数 / `[1]` birads | `[1, nc]` / `[1, nb]`（批量首维 N） | FP32 |
| `OCRReco` | `GenericPostParams` | `[0]` 长度 / `[1]` argmax | 长度 `[N]`；argmax `[N, seq]`（CTC 折叠） | INT64 |
| `SemanticSeg` | `ConfidenceFilterParams` | `[0]` 类别图 | `[1, nc, h, w]`（批量 `[N, nc, h, w]`） | FP32 |
| `UNetDualOutputSeg` | `GenericPostParams` | `[0]` prob / `[1]` mask | 各 `[1, w, h]`（decoder 读 `shape[2]=h`, `shape[1]=w`） | FP32 |
| `RawModelOutput` | `GenericPostParams` | 全部原样透传 | 任意 | 任意 |

- `nc` = 类别数，`nb` = birads 数，`anchors` = 锚点数，`seq` = 序列长。
- `AnchorDetParams` 需 `condThre` + `nmsThre`；`ConfidenceFilterParams` 需
  `condThre`；`GenericPostParams` 只需 `outputNames`。
- 检测类输出的坐标还原依赖预处理写入的 `FrameTransformContext`，缺失即返回
  `InferInvalidInput`。

## 4. 预处理契约

内置帧预处理插件（`CpuGenericPreprocess` / `CudaGenericPreprocess` /
`FrameWithMaskPreprocess`）消费 `FramePreprocessArg`，产出单个模型输入张量，名字
取 `inputNames[0]`，shape 依 `hwc2chw` 为 `{N,C,H,W}` 或 `{N,H,W,C}`。
`FrameWithMaskPreprocess` 把 mask 光栅化为额外通道，**调用方必须把
`inputShape.c` 设为含 mask 的真实通道数**（3 图 + 1 mask = 4）。

## 5. 新增一个检测头的最小步骤

1. 新建 `src/postproc/my_det.{hpp,cpp}`，继承 `FramePostprocBase<AnchorDetParams,
   true>`（`true` = 需要预处理变换上下文做坐标还原）。
2. 实现 `processTyped` / `batchProcessTyped` 两个纯虚 hook。
3. 在 .cpp 里 `REGISTER_POSTPROCESS_ALGO(MyDet);`。
4. JSON 配置里 `types.postproc` 填 `"MyDet"`，`postprocParams` 给
   `condThre/nmsThre/outputNames`。

不需要改任何工厂或分发代码。
