#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

__host__ void check(cudaError_t err, const char* const func, const char* const file, const int line);
__host__ void checkLast(const char* const file, const int line);
