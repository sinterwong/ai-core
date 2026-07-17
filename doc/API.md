# AI Core API 参考

API 按使用频率自上而下组织。先看「核心入口」一节跑通一次推理，再按需查「数据结构」「枚举」「插件开发」。

所有类型位于 `ai_core` 或 `ai_core::dnn` 命名空间。

## 1. 核心入口

### `ai_core::dnn::AlgoInference`

三段流水线的统一入口，定义于 `<ai_core/algo_inference.hpp>`。

```cpp
class AlgoInference {
public:
  AlgoInference(const AlgoModuleTypes& module_types,
                const AlgoInferParams& infer_params);
  ~AlgoInference();

  InferErrorCode initialize();
  InferErrorCode terminate();

  InferErrorCode infer(const AlgoInput& input,
                       const AlgoPreprocParams& preproc_params,
                       const AlgoPostprocParams& postproc_params,
                       AlgoOutput& output);

  InferErrorCode batchInfer(const std::vector<AlgoInput>& inputs,
                            const AlgoPreprocParams& preproc_params,
                            const AlgoPostprocParams& postproc_params,
                            std::vector<AlgoOutput>& outputs);

  const ModelInfo& getModelInfo() const noexcept;
  const AlgoModuleTypes& getModuleTypes() const noexcept;
};
```

调用顺序：`initialize` → 多次 `infer` / `batchInfer` → `terminate`。未初始化就调用 `infer` 会得到 `InferErrorCode::NotInitialized`。

### `ai_core::dnn::AlgoManager`

进程内管理多个 `AlgoInference` 实例：

```cpp
class AlgoManager {
public:
  InferErrorCode registerAlgo(const std::string& name,
                              const std::shared_ptr<AlgoInference>& algo);
  InferErrorCode unregisterAlgo(const std::string& name);

  InferErrorCode infer(const std::string& name,
                       AlgoInput& input,
                       AlgoPreprocParams& preproc_params,
                       AlgoOutput& output,
                       AlgoPostprocParams& postproc_params);

  std::shared_ptr<AlgoInference> getAlgo(const std::string& name) const;
  bool hasAlgo(const std::string& name) const;
  void clear();
};
```

`AlgoManager` 不可拷贝，可移动。

### 单组件直接使用

如果不想要 `AlgoInference` 的串联，也可以单独用：

- `ai_core::dnn::AlgoPreproc(const std::string& module_name)`
- `ai_core::dnn::AlgoInferEngine(const std::string& module_name, const AlgoInferParams& infer_params)`
- `ai_core::dnn::AlgoPostproc(const std::string& module_name)`

三者的接口分别是 `process / batchProcess`、`infer`、`process / batchProcess`，全部需要先 `initialize()`。

## 2. 配置与参数

### `AlgoModuleTypes`

```cpp
struct AlgoModuleTypes {
  std::string preproc_module;
  std::string infer_module;
  std::string postproc_module;
};
```

三个字段是已注册插件的名字。

### `AlgoInferParams`

```cpp
struct AlgoInferParams {
  std::string name;
  std::string model_path;
  bool need_decrypt = false;
  std::string decryptkey_str;
  DeviceType device_type;
  DataType   data_type;
  std::map<std::string, size_t> max_output_buffer_sizes;
};
```

`max_output_buffer_sizes` 用来预分配输出缓冲，键是张量名。GPU 后端推荐填，CPU 后端可以空着。

### `AlgoPreprocParams` / `AlgoPostprocParams`

每次 `infer` 调用都传一份，覆盖一些按需调整的参数。底层是 `ParamCenter<std::variant<...>>`，详见第 4 节。

## 3. 输入 / 输出结构

### 输入

```cpp
struct FrameInput {
  std::shared_ptr<cv::Mat> image;
  std::shared_ptr<cv::Rect> input_roi;  // 可选，nullptr 表示全图
};

struct FrameInputWithMask {
  FrameInput frame_input;
  std::vector<cv::Rect> mask_regions;
};
```

### 输出

```cpp
struct BBox {
  cv::Rect rect;
  float score;
  int   label;
};

struct ClsRet { float score; int label; };

struct FprClsRet {
  float score;
  int   label;
  int   birad;
  std::vector<float> score_probs;
};

struct DetRet { std::vector<BBox> bboxes; };

struct SegRet { std::map<int, std::vector<Contour>> cls_to_contours; };

struct DualRawSegRet {
  std::shared_ptr<cv::Mat> mask;
  std::shared_ptr<cv::Mat> prob;
  std::shared_ptr<cv::Rect> roi;
  float ratio;
  int   left_shift;
  int   top_shift;
};

struct OCRRecoRet {
  int64_t output_lengths;
  std::vector<int64_t> outputs;
};

using RawModelOutput = TensorData;
```

`Contour` 是 `std::vector<cv::Point>`，定义在 `<ai_core/common_types.hpp>`。

## 4. 参数包装：`ParamCenter<T>`

```cpp
template <typename P> class ParamCenter {
public:
  template <typename T> void setParams(T params);
  template <typename T> T* getParams();
  template <typename T> const T* getParams() const;
  template <typename Func> void visitParams(Func&& func);
};
```

`setParams<T>()` 会覆盖之前的值。`getParams<T>()` 返回 `nullptr` 表示当前装的是别的类型。

## 5. 内存：`TypedBuffer`

定义于 `<ai_core/typed_buffer.hpp>`。统一管理 CPU 内存、GPU 显存、Pinned Host 内存。

### 创建

```cpp
static TypedBuffer createFromCpu(DataType type, const std::vector<uint8_t>& data);
static TypedBuffer createFromCpu(DataType type, std::vector<uint8_t>&& data);

static TypedBuffer createFromCpuRef(DataType type, const void* host_ptr,
                                    size_t size_bytes, bool manage_memory = false);

static TypedBuffer createFromGpu(DataType type, size_t size_bytes, int device_id = 0);
static TypedBuffer createFromGpu(DataType type, void* device_ptr,
                                 size_t size_bytes, int device_id, bool manage_memory);

static TypedBuffer createPinnedHost(DataType type, size_t size_bytes);
```

`createFrom*Ref` 是零拷贝包装，缓冲区生命周期由外部管理。`manage_memory=true` 时 `TypedBuffer` 在析构时释放该指针。

### 访问

```cpp
template <typename T> const T* getHostPtr() const;
template <typename T> T*       getHostPtr();
const void* getRawHostPtr() const;
void*       getRawHostPtr();
void*       getRawDevicePtr() const;
```

`getHostPtr<T>()` 在位置不是 CPU、或 `sizeof(T)` 与 `dataType` 不匹配时抛 `std::runtime_error`。

### 元信息

```cpp
DataType        dataType() const noexcept;
BufferLocation  location() const noexcept;   // CPU / GpuDevice
BufferMemoryType memoryType() const noexcept; // Pageable / Pinned / Managed
size_t getSizeBytes()   const noexcept;
size_t getElementCount() const noexcept;
int    getDeviceId()    const noexcept;
bool   isPinned()       const noexcept;
bool   isReference()    const noexcept;
```

### 修改

```cpp
void resize(size_t new_element_count);  // CPU 保留数据，Pinned/GPU 重新分配
void clear();
void setCpuData(DataType type, const std::vector<uint8_t>& data);
void setGpuDataReference(DataType type, void* ptr, size_t size_bytes, int dev_id = 0);
```

## 6. 张量集合：`TensorData`

```cpp
struct TensorData {
  std::map<std::string, TypedBuffer> datas;
  std::map<std::string, std::vector<int>> shapes;
};
```

应用层一般不需要直接构造 `TensorData`，由预处理、推理、后处理插件在内部维护。

## 7. 上下文：`DataPacket`

```cpp
struct DataPacket {
  DataPacketId id;
  std::map<std::string, std::any> params;

  template <typename T> T getParam(const std::string& key) const;
  template <typename T> std::optional<T> getOptionalParam(const std::string& key) const;
  template <typename T> void setParam(const std::string& key, T value);
  bool has(const std::string& key) const;
};
```

`getParam` 在 key 不存在或类型不匹配时抛 `std::runtime_error`。框架里 `AlgoConstructParams` 是 `DataPacket` 的别名；`RuntimeContext` 是独立结构，预处理→后处理的变换信息走类型化槽位：

```cpp
struct RuntimeContext {
  std::optional<FrameTransformContext> frame_transform;       // 单帧
  std::vector<FrameTransformContext> frame_transform_batch;   // batch
  DataPacket extras;                                          // 自由扩展
};
```

## 8. 异步接口

### `IAsyncInferEngine`

后端支持时，推理引擎插件可以额外实现 `IAsyncInferEngine`（继承自 `IInferEnginePlugin`）：

```cpp
class IAsyncInferEngine : public IInferEnginePlugin {
public:
  virtual std::shared_ptr<IExecutionContext> createExecutionContext() = 0;
  virtual TypedBuffer allocateAcceleratorBuffer(DataType type, size_t size_bytes) = 0;

  struct ContextPackage {
    std::shared_ptr<IExecutionContext> context;
    TensorData inputs;
    TensorData outputs;
  };
  virtual ContextPackage createContextPackage();
  virtual std::vector<std::shared_ptr<IExecutionContext>> createContextPool(size_t count);
};
```

`allocateAcceleratorBuffer` 在不同后端下行为不同：CUDA/HIP 下分配 Pinned Host 内存；集成 GPU/NPU 下可能分配 Shared Memory；纯 CPU 下返回对齐内存。`ContextPackage` 用于 Zero-Copy 或 CUDA Graph 这类输入/输出地址不能变的场景。

## 9. 枚举

### `InferErrorCode`

```cpp
enum class InferErrorCode : int32_t {
  SUCCESS = 0,

  InitFailed = 100,
  InitConfigFailed = 101,
  InitModelLoadFailed = 102,
  InitDeviceFailed = 103,
  InitMemoryAllocFailed = 104,
  InitDecryptionFailed = 105,
  NotInitialized = 106,
  InitRuntimeFailed = 107,
  InitEngineFailed = 108,
  InitContextFailed = 109,
  InitBindingFailed = 110,

  InferFailed = 200,
  InferInputError = 201,
  InferOutputError = 202,
  InferDeviceError = 203,
  InferPreprocessFailed = 204,
  InferMemoryError = 205,
  InferSetInputFailed = 206,
  InferExtractFailed = 207,
  InferUnsupportedOutputType = 208,
  InferTypeMismatch = 209,
  InferSizeMismatch = 210,
  InferInvalidInput = 211,
  InferExecutionFailed = 212,
  InferBindingError = 213,

  StreamCreationFailed = 250,
  StreamSyncFailed = 251,
  GraphCaptureFailed = 252,
  GraphLaunchFailed = 253,
  AsyncOperationPending = 254,

  TerminateFailed = 300,

  AlgoNotFound = 400,
  AlgoRegisterFailed = 401,
  AlgoUnregisterFailed = 402,
  AlgoInferFailed = 403,
};
```

### `DeviceType`

```cpp
enum class DeviceType { CPU = 0, GPU = 1 };
```

### `DataType`

```cpp
enum class DataType : uint8_t {
  FLOAT32 = 0, FLOAT16, INT32, INT64, INT8,
};
```

### `BufferLocation` / `BufferMemoryType`

```cpp
enum class BufferLocation  { CPU, GpuDevice };
enum class BufferMemoryType { Pageable, Pinned, Managed };
```

## 10. 模型信息

```cpp
struct ModelInfo {
  std::string name;

  struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    DataType data_type;
  };

  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
};
```

通过 `AlgoInference::getModelInfo()` 拿到，能用来核对当前模型的输入输出是否和你的代码假设一致。

## 11. 插件开发

### 接口

```cpp
class IPreprocessPlugin {
public:
  virtual ~IPreprocessPlugin() = default;
  virtual InferErrorCode process(const AlgoInput&,
                       const AlgoPreprocParams&,
                       TensorData&,
                       std::shared_ptr<RuntimeContext>&) const = 0;
  virtual InferErrorCode batchProcess(const std::vector<AlgoInput>&,
                            const AlgoPreprocParams&,
                            TensorData&,
                            std::shared_ptr<RuntimeContext>&) const = 0;
};

class IPostprocessPlugin {
public:
  virtual ~IPostprocessPlugin() = default;
  virtual InferErrorCode process(const TensorData&,
                       const AlgoPostprocParams&,
                       AlgoOutput&,
                       std::shared_ptr<RuntimeContext>&) const = 0;
  virtual InferErrorCode batchProcess(const TensorData&,
                            const AlgoPostprocParams&,
                            std::vector<AlgoOutput>&,
                            std::shared_ptr<RuntimeContext>&) const = 0;
};

class IInferEnginePlugin {
public:
  virtual InferErrorCode initialize() = 0;
  virtual InferErrorCode infer(const TensorData& inputs, TensorData& outputs) = 0;
  virtual InferErrorCode terminate() = 0;
  virtual const ModelInfo& getModelInfo() = 0;
  virtual void prettyPrintModelInfos();
};
```

注意：预处理和后处理插件用 `bool` 表示成功失败，推理引擎插件用 `InferErrorCode`。

### 注册

```cpp
#include "ai_core/plugin_registrar.hpp"

REGISTER_PREPROCESS_ALGO(MyPreproc);
REGISTER_INFER_ENGINE(MyEngine);
REGISTER_POSTPROCESS_ALGO(MyPostproc);
```

宏把字符串 `"MyPreproc"` 等与默认构造函数 / `AlgoConstructParams` 构造函数绑定。默认注册在 `src/registrar/` 下，链接主库即可生效。注册自己的插件时，把上述宏放到对应 `.cpp` 末尾即可。

### 内置插件一览

| 阶段 | 名字 | 用途 |
| --- | --- | --- |
| 预处理 | `CpuGenericPreprocess` | 单帧图通用预处理（OpenCV CPU） |
| 预处理 | `CudaGenericPreprocess` | 单帧图通用预处理（CUDA，需 `WITH_TRT_ENGINE`） |
| 预处理 | `FrameWithMaskPreprocess` | 带掩码区域的图（CPU） |
| 推理 | `OrtAlgoInference` | ONNX Runtime |
| 推理 | `NCNNAlgoInference` | NCNN |
| 推理 | `TrtAlgoInference` | TensorRT |
| 后处理 | `Yolov11Det` / `RTMDet` / `NanoDet` | 锚框检测，参数 `AnchorDetParams` |
| 后处理 | `SoftmaxCls` / `FprCls` / `RawModelOutput` / `OCRReco` / `UNetDualOutputSeg` | 通用后处理，参数 `GenericPostParams` |
| 后处理 | `SemanticSeg` | 语义分割，参数 `ConfidenceFilterParams` |

算法由插件名直接选择：`AlgoModuleTypes::postproc_module` 填上表中的名字即可，参数结构里不再有 `algo_type` 字段。

## 12. 日志

`<ai_core/logger.hpp>` 提供流式和格式化两种用法：

```cpp
LOG_INFO_S << "load model: " << path;
LOG_ERROR_FMT("infer failed, code = %d", static_cast<int>(code));
```

日志级别（`LogLevel`）：`Trace / Debug / Info / Warning / Error / Fatal / Off`。运行时通过 `ai_core::logging::Logger::instance().setLevel(...)` 调整。`LOG_*` 宏在编译期会按 `AI_CORE_LOG_LEVEL` 过滤，不会把禁用级别的字符串拼进二进制。
