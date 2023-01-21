#include <thrust/reduce.h>
#include <thrust/device_malloc.h>

#include <iostream>
#include "../common/parser.h"

#include "errors.cuh"
#include "kernels.cuh"


struct compare
{
  __host__ __device__
  bool operator()(float lhs, float rhs)
  {
  return fabsf(lhs) < fabsf(rhs);
  }
};

void encodeDevice(const ProgramOptions& options);
const char* encodeFile(int xComponents, int yComponents, const char* filename, float* deviceAc, char* encodedACValuesBuffer);
const char* encodePixels(uint8_t* data, int xComponents, int yComponents, int width, int height, float* deviceAc, char *encodedACValuesBuffer);
void calculateFactor(uint8_t* pixels, int xComponent, int yComponent, int width, int height, float result[3], float* vec_r, float* vec_g, float* vec_b);
