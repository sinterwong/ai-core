# ai_core starter

一个可直接复制起手的最小产品骨架：`find_package(ai_core)` → JSON 配置加载 →
一次推理 → 打印结果。约 80 行 C++，无框架内部依赖。

## 用它起一个新产品

1. 把整个 `starter/` 目录复制出去。
2. 换掉 `conf/*.json` 里的 `modelPath` 和参数，指向你的模型。
3. `cmake` + `build`，跑起来。

## 构建（消费已安装的 ai_core）

```bash
cmake -B build \
  -DCMAKE_PREFIX_PATH="/path/to/ai_core/install/share;/path/to/ai_core/install;/path/to/opencv"
cmake --build build
```

`ai_core-config.cmake` 会通过 `find_dependency` 自动带出 OpenCV，所以一次
`find_package(ai_core)` 足够。

**后端运行库**：`ai_core` 编译进了推理后端（ONNX Runtime / TensorRT），其共享库
需要在链接与运行时可达。开发树里它们在 `3rdparty/target/.../{onnxruntime,tensorrt}/lib`：

```bash
# 链接期让 ld 找到后端库（也可把后端库装到系统库路径）
-DCMAKE_EXE_LINKER_FLAGS="-L<onnxruntime>/lib -Wl,-rpath,<onnxruntime>/lib"
# 运行期
export LD_LIBRARY_PATH=<ai_core>/install/lib:<onnxruntime>/lib:$LD_LIBRARY_PATH
```

生产部署时建议把后端库与 `ai_core` 一起打包安装（见 `scripts/bootstrap.sh`）。

## 运行

```bash
./build/ai_core_starter conf/yolo_det_ort.json <image.png>
# detections: 2
#   label=7 score=0.54 rect=[...]
#   label=0 score=0.80 rect=[...]
```

`conf/yolo_det_ort.json` 里 `modelPath` 相对配置文件的祖父目录解析
（`<root>/conf/x.json` → `<root>/models/...`），与 `assets/` 布局一致。
