# AI Core

AI Core is a C++ library designed for managing and executing AI inference algorithms. It features a modular architecture, leveraging registrar patterns (`AlgoRegistrar`, `PostprocessRegistrar`) and factory mechanisms (`AlgoInferFactory`, `PostprocFactory`). This design promotes extensibility, allowing developers to easily register and integrate new algorithms and vision processing modules. The library appears to support various inference engines (ONNX Runtime is used in CI examples) and is geared towards computer vision applications such as object detection and tracking.

## Build and Installation

This project uses CMake for building and managing dependencies. The following instructions are based on the GitHub Actions CI workflow.

### Prerequisites

*   A C++ compiler supporting C++17 (GCC 13 is used in CI).
*   CMake (version 3.16 or higher recommended).
*   Ninja (optional, but used in CI for faster builds).
*   Git LFS (for handling large model files).

### Dependencies

The project relies on several third-party libraries. The CI workflow downloads prebuilt dependencies from a specific URL. For a local build, you might need to ensure these dependencies are available or adjust the CMake configuration. The core dependencies seem to include:
*   ONNX Runtime (for model inference)
*   OpenCV (likely for image processing)
*   Other utility libraries (encryption, logging)

### Building the Project

1.  **Clone the repository (with submodules and LFS):**
    ```bash
    git clone --recurse-submodules <repository_url>
    cd <repository_name>
    git lfs pull
    ```

2.  **Download and Extract Dependencies (if not handled by CMake automatically):**
    The CI script downloads dependencies from `https://github.com/sinterwong/ai-core/releases/download/v1.0.0-alpha/dependency-Linux_x86_64.tgz` and extracts them to `3rdparty/target/`. You may need to replicate this step or configure CMake to find your local installations of these dependencies.

3.  **Configure CMake:**
    Create a build directory and run CMake.
    ```bash
    mkdir build
    cd build
    cmake .. -GNinja \
          -DCMAKE_INSTALL_PREFIX=../install \
          -DBUILD_AI_CORE_TESTS=ON \
          -DINFER_ENGINE=ORT \
          -DCMAKE_BUILD_TYPE=Release
    ```
    *   `CMAKE_INSTALL_PREFIX`: Specifies the installation directory.
    *   `BUILD_AI_CORE_TESTS=ON`: Enables building of tests.
    *   `INFER_ENGINE=ORT`: Specifies ONNX Runtime as the inference engine. You might need to change this if using a different engine or if the project supports multiple.
    *   `CMAKE_BUILD_TYPE=Release`: Specifies the build type.

4.  **Build the project:**
    ```bash
    cmake --build . --config Release
    ```
    Or if using Ninja directly:
    ```bash
    ninja
    ```

5.  **Install (optional):**
    This step will copy the built libraries, executables, and headers to the directory specified by `CMAKE_INSTALL_PREFIX`.
    ```bash
    cmake --install .
    ```

### Running Tests

After building and installing, you can run the tests. The CI script sets up `LD_LIBRARY_PATH` before running tests. You might need to do something similar depending on your system and where the dependencies are located.

From the `install` directory (or wherever you installed the project):
```bash
# Example of setting up library path (adjust paths as needed)
export LD_LIBRARY_PATH=./lib:../build/3rdparty/encryption-tool/x86_64/lib:../build/3rdparty/logger/x86_64/lib:<path_to_onnxruntime>/lib:<path_to_opencv>/lib:$LD_LIBRARY_PATH

./tests/ai_core_tests --gtest_filter=*.*
```

This README provides a basic overview. For more detailed information, please refer to the project's wiki (if available) or source code documentation.
