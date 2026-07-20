# Benchmark 基线存档

每个版本的 benchmark 结果存档于此，供后续版本前后对比。性能类改动必须对照上一版基线，数据写进提交信息。

## 采集方式

```bash
cmake -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_AI_CORE_BENCHMARKS=ON -DWITH_ORT_ENGINE=ON \
  -DWITH_NCNN_ENGINE=ON -DWITH_TRT_ENGINE=ON
cmake --build build-release -j && cmake --install build-release

cd install-release
LD_LIBRARY_PATH=$PWD/lib:<3rdparty-libs> ./benchmarks/ai_core_benchmarks \
  --benchmark_filter='BM_CPU_FramePreproc_Yolo|BM_CPU_YoloDetPostproc|GPU_FramePreproc|YoloInfer' \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
  --benchmark_format=json --benchmark_out=../benchmarks/baseline/v1.5-x86_64.json
```

- **Release** 构建（Debug 计时无意义）。
- 结果受机器负载影响；对比时取同机、空载下的 median。

## 环境（v1.5 基线）

- CPU：16×2304 MHz，L3 24 MiB
- 后端：ONNX Runtime 1.20.1、NCNN、TensorRT 10.14；OpenCV 4.10
- 平台：Linux x86_64

## v1.5 关键数字（median，YOLOv11n 640×640 单帧）

| benchmark | median (ms) | 说明 |
|---|---|---|
| `BM_CPU_FramePreproc_Yolo` | **3.16** | **v1.6 靶子：CPU 单帧预处理下降 ≥40% → ≤1.90 ms** |
| `BM_GPU_FramePreproc_Yolo` | 1.47 | CUDA 预处理（HWC + FP16） |
| `BM_CPU_YoloDetPostproc` | 1.45 | 检测后处理解码 + NMS |
| `BM_ORT_CPU_DATA_YoloInfer` | 65.7 | ORT CPU 前向 |
| `BM_NCNN_CPU_DATA_YoloInfer` | 72.6 | NCNN CPU 前向 |

完整数据见 `v1.5-x86_64.json`。TRT 异步/CUDA Graph 系列（`BM_TRT_*`）是 v1.7 并发工作的基线，届时单独采集。
