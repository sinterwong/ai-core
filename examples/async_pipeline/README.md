# 异步流水线示例（async_pipeline）

框架的性能卖点：从公共 API 一路走到 context pool + pinned buffer + CUDA Graph 的完整异步流水线。

## 依赖

- `WITH_TRT_ENGINE=ON`（异步能力由 TensorRT 后端提供）。
- 一个 TensorRT 引擎：`scripts/fetch_models.sh --trt-only` 从 ONNX 重建。

## 运行

```bash
cd install
LD_LIBRARY_PATH=$PWD/lib:<3rdparty-libs> \
  ./bin/ai_core_example_async_pipeline [engine_path] [num_workers] [total_frames]
# 默认: assets/models/yolov11n_trt_fp16.engine 4 400
```

输出形如：`Processed 400/400 frames in 0.74 s (543 img/s, 4 workers)`。

## 它演示了什么

```
AlgoInferEngine::getAsyncEngine()   // 异步正门，无需 dynamic_cast 插件
  -> createContextPool(N)           // N 个 IExecutionContext，各自独立 CUDA stream
  -> allocateAcceleratorBuffer()    // pinned host 内存，H2D/D2H 可异步重叠
  -> setGraphEnabled(true)          // CUDA Graph，摊薄 kernel launch 开销
  -> N worker 线程各持一个 context  // context 非线程安全，一线程一 context
     从共享帧队列取活，inferAsync + synchronize
```

CUDA Graph + pinned buffer 把单 GPU 吞吐显著抬高（本机 RTX 3060 Laptop、
yolov11n@640-fp16：同步 ~319 img/s → 异步流水线 ~543 img/s）。真实场景里
`makePinnedInput` 的位置就是预处理直接写 accelerator buffer 的落点。

线程模型见 `doc/Framework.md`：engine 共享、可并发建 context；每个
`IExecutionContext` 非线程安全，由单个 worker 独占。
