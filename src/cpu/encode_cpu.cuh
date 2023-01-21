#include <stdint.h>
#include <stdlib.h>

#include "../common/parser.h"


void encodeCPU(const ProgramOptions& options);
const char *blurHashForFile(int xComponents, int yComponents,const char *filename);
const char *blurHashForPixels(int xComponents, int yComponents, int width, int height, uint8_t *rgb, size_t bytesPerRow);
float *multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t *rgb, size_t bytesPerRow);
int encodeAC(float r, float g, float b, float maximumValue);
float sRGBToLinear(int value);
float signCPUPow(float value, float exp);