#ifndef CUDA_UTILS_CUDA_HELPER_CUH
#define CUDA_UTILS_CUDA_HELPER_CUH
#include <cuda_runtime.h>
#include <stdexcept>

class CudaError : public std::runtime_error {
public:
  CudaError(const char *msg, const char *func, const char *file, int line)
      : std::runtime_error(std::string(msg) + " at " + func + " (" + file +
                           ":" + std::to_string(line) + ")") {}
};

#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char *func, const char *file,
                           const int line) {
  if (err != cudaSuccess) {
    throw CudaError(cudaGetErrorString(err), func, file, line);
  }
}

#endif