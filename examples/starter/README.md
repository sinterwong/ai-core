# ai_core starter

一个可复制到产品仓库的最小安装包消费工程：`find_package(ai_core)`、从安装目录
动态加载插件、读取 JSON 配置、执行一次推理。

## 构建

先安装包含 config、OpenCV pre/postproc 和 ONNX Runtime 插件的 ai-core：

```bash
scripts/deps.sh init config onnxruntime
cmake --preset developer
cmake --build --preset developer
cmake --install build/developer
```

再把 `starter/` 复制到仓库外，指向该安装前缀：

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/ai-core/install/developer
cmake --build build
```

starter 会请求 package components `core`、`config`、
`plugin_preproc_opencv`、`plugin_postproc_opencv`、`plugin_onnxruntime`；缺少任意
组件会在 configure 阶段失败。OpenCV 仅用于本示例的图片 I/O，因此由 starter
自己 `find_package(OpenCV)`。

运行时还需让系统动态加载器找到 ONNX Runtime SDK；ai-core 和官方插件路径由
安装包自身处理。

```bash
export LD_LIBRARY_PATH=/path/to/ai-core/.deps/Linux_x86_64/onnxruntime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
./build/ai_core_starter conf/yolo_det_ort.json <image.png>
```

配置里的 `modelPath` 相对配置文件的祖父目录解析
（`<root>/conf/x.json` → `<root>/models/...`），与 `assets/` 布局一致。
