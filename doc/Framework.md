# AI Core 框架设计

本文说明 AI Core 的整体结构、关键数据结构和扩展方式。读完后你应该能上手用 `AlgoInference`，并知道在哪里添加自定义的预处理、推理后端或后处理。

## 流水线

一次推理分为三段：

1. **预处理**（`AlgoPreproc`）—— 把 `AlgoInput` 里的原始数据（例如 `cv::Mat`）转成模型要的 `TensorData`。
2. **推理**（`AlgoInferEngine`）—— 调用底层后端（ONNX Runtime、NCNN、TensorRT）做前向，输出 `TensorData`。
3. **后处理**（`AlgoPostproc`）—— 把模型的原始张量解析成业务结果，写入 `AlgoOutput`。

`AlgoInference` 把这三段串起来，对外只暴露 `initialize / infer / batchInfer / terminate` 等少量方法。

```
AlgoInput -> AlgoPreproc -> TensorData -> AlgoInferEngine -> TensorData -> AlgoPostproc -> AlgoOutput
```

`RuntimeContext` 在 `infer` 调用的整个生命周期里向下游传递。预处理把变换信息（原始尺寸、缩放系数、padding 等）写入类型化槽位 `frame_transform` / `frame_transform_batch`（`FrameTransformContext`），后处理读出来做坐标还原；自定义插件的自由扩展数据放 `extras`（`DataPacket`）。

## 关键数据结构

### `AlgoInput` / `AlgoOutput`

`AlgoInput` 当前可装 `FrameInput`（单帧图）或 `FrameInputWithMask`（带掩码区域）。`AlgoOutput` 可装 `ClsRet`、`DetRet`、`SegRet`、`DualRawSegRet`、`OCRRecoRet`、`FprClsRet`、`RawModelOutput` 等。

它们都是 `ParamCenter<std::variant<...>>`。用 `setParams<T>()` 存，用 `getParams<T>()` 取。类型不对就拿不到指针，不需要写 type cast。

### `TensorData`

```cpp
struct TensorData {
  std::map<std::string, TypedBuffer> datas;
  std::map<std::string, std::vector<int>> shapes;
};
```

流水线内部按张量名传递数据，与 ONNX Runtime、TensorRT 的输入输出约定一致。应用层一般不需要直接构造 `TensorData`，由插件负责。

### `TypedBuffer`

带类型的内存缓冲区，能同时表示 CPU 内存、GPU 显存和 Pinned Host 内存。提供：

- 静态工厂：`createFromCpu`、`createFromGpu`、`createPinnedHost`、`createFromCpuRef`、`createFromGpuRef`
- `getHostPtr<T>()` / `getRawHostPtr()` / `getRawDevicePtr()`
- 元信息：`dataType`、`location`、`memoryType`、`getSizeBytes`、`getElementCount`

`getHostPtr<T>()` 会校验当前位置是 CPU，且 `sizeof(T)` 与 `dataType` 匹配。位置或类型不匹配时直接抛异常，避免静默错误。

### `DataPacket`

`std::map<std::string, std::any>` 的薄封装，用 `setParam` / `getParam` / `getOptionalParam` / `has` 访问。框架里两处用到：

- `AlgoConstructParams`：插件构造时的配置
- `RuntimeContext`：单次推理的上下文

### `AlgoModuleTypes` 与 `AlgoInferParams`

```cpp
struct AlgoModuleTypes {
  std::string preproc_module;   // 注册过的预处理插件名
  std::string infer_module;     // 注册过的推理后端名
  std::string postproc_module;  // 注册过的后处理插件名
};

struct AlgoInferParams {
  std::string name;             // 算法名（自己取，用于日志/识别）
  std::string model_path;       // 模型文件路径
  bool need_decrypt = false;    // 模型是否加密
  std::string decryptkey_str;   // 解密密钥
  DeviceType device_type;       // CPU / GPU
  DataType   data_type;         // 推理精度
  std::map<std::string, size_t> max_output_buffer_sizes;  // 预分配输出缓冲
};
```

## 三个核心组件

每个组件都通过工厂按名字创建具体插件，对外只暴露一个 `process` / `infer` 入口。

### `AlgoPreproc`

构造时传插件名。`initialize()` 阶段从 `PreprocFactory` 取出对应实现。`process()` 内部把任务转给该插件。

内置：`CpuGenericPreprocess`（单帧，OpenCV CPU）、`CudaGenericPreprocess`（单帧，CUDA，需 `WITH_TRT_ENGINE`）、`FrameWithMaskPreprocess`（带掩码，CPU）。执行设备由插件名决定，不再有运行期分发字段。

参数见 `FramePreprocessArg`：

- `model_input_shape` — `{w, h, c}`
- `need_resize`、`is_equal_scale`、`pad`
- `mean_vals`、`norm_vals`
- `hwc2chw`
- `data_type`、`output_location` — 输出放 CPU 还是 GPU
- `input_names` — 多输入模型时给出每个输入的名字

### `AlgoInferEngine`

构造时传插件名 + `AlgoInferParams`。`infer()` 接收 `TensorData`、返回 `TensorData`。

内置：`OrtAlgoInference`（ONNX Runtime）、`NCNNAlgoInference`（NCNN）、`TrtAlgoInference`（TensorRT）。

`getModelInfo()` 返回当前模型的输入/输出张量名、shape、dtype。

### `AlgoPostproc`

构造时传插件名。`process()` 把模型输出解析成 `AlgoOutput` 中的一种类型。

内置（每个算法就是一个插件，按名字直接注册到工厂）：

- `Yolov11Det` / `RTMDet` / `NanoDet` —— 锚框检测家族，参数 `AnchorDetParams`
- `SoftmaxCls` / `FprCls` / `RawModelOutput` / `OCRReco` / `UNetDualOutputSeg` —— 通用家族，参数 `GenericPostParams`
- `SemanticSeg` —— 置信度过滤家族，参数 `ConfidenceFilterParams`

参数：

- `AnchorDetParams`：`cond_thre`、`nms_thre`、`output_names`
- `GenericPostParams`：`output_names`
- `ConfidenceFilterParams`：`cond_thre`、`output_names`

新增一个后处理算法 = 一个继承 `FramePostprocBase<ParamsT, RequiresPrepContext>` 的新 `.cpp` + 一行注册宏。

## 工厂与注册

插件在编译期通过宏挂到三个全局工厂里：

- `PreprocFactory = Factory<IPreprocessPlugin>`
- `InferEngineFactory = Factory<IInferEnginePlugin>`
- `PostprocFactory = Factory<IPostprocessPlugin>`

注册宏：

```cpp
REGISTER_PREPROCESS_ALGO(MyPreproc);
REGISTER_INFER_ENGINE(MyEngine);
REGISTER_POSTPROCESS_ALGO(MyPostproc);
```

宏的副作用是把字符串 `"MyPreproc"` 和对应的构造器放进工厂的 map。`AlgoModuleTypes` 写哪个名字，就调用哪个。

默认注册在 `src/registrar/` 下，链接库时自动生效。

## 自定义插件

要加新的算法或后端，按以下步骤：

1. 继承 `IPreprocessPlugin`、`IInferEnginePlugin` 或 `IPostprocessPlugin`，实现纯虚函数。
2. 在 `.cpp` 末尾用对应宏注册。
3. 链接到主程序。
4. 在 `AlgoModuleTypes` 里写新插件的名字。

预处理和后处理插件的构造函数默认无参（宏里用 `std::make_shared<AlgoName>()`）。推理引擎插件的构造函数带 `const AlgoConstructParams&`（宏里用 `std::make_shared<EngineName>(cparams)`），可以在构造时读 `DataPacket` 拿到配置。

## `AlgoManager`

如果一个进程里要跑多种算法，用 `AlgoManager` 集中管理：

```cpp
dnn::AlgoManager mgr;
mgr.registerAlgo("det", std::make_shared<dnn::AlgoInference>(detModules, detParams));
mgr.registerAlgo("cls", std::make_shared<dnn::AlgoInference>(clsModules, clsParams));

mgr.infer("det", input, preproc_params, output, postproc_params);
```

注册失败会返回 `InferErrorCode::AlgoRegisterFailed` / `AlgoNotFound`。

## 异步与 Zero-Copy

后端支持时，推理引擎插件可以实现 `IAsyncInferEngine`：

- `createExecutionContext()` —— 拿到一个轻量的执行上下文（对应 CUDA stream、TensorRT execution context 等）。
- `allocateAcceleratorBuffer(type, bytes)` —— 申请适合该后端的 host 内存（CUDA 下是 Pinned）。
- `createContextPackage()` / `createContextPool(n)` —— 预绑定缓冲区，适合 CUDA Graph 这类地址不能变的场景。

`TypedBuffer` 已经统一了 CPU 内存、Pinned Host 内存和 GPU 显存，所以预分配的输入/输出可以直接在流水线里流转。
