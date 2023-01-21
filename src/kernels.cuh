#include <vector>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/device_malloc.h>

extern __constant__ char charactersDevice[84] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

__global__ void encodeACValues(float *acBufferDevice, char* encodedACValuesBuffer, float maximumValue, int n);
__global__ void calculatePixelColors(uint8_t* pixels, float* vec_r, float* vec_g, float* vec_b, int xComponent, int yComponent, int width, int height);
__device__ void encodeIntDevice(char *buffer, int index, int number);
__device__ float sRGBToLinear(uint8_t value);
__device__ float signPow(float value, float exp);
