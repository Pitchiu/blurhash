#include<math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int linearTosRGB(float value) {
	float v = fmaxf(0, fminf(1, value));
	if (v <= 0.0031308) return v * 12.92 * 255 - 0.5;
	else return (1.055 * powf(v, 1 / 2.4) - 0.055) * 255 - 0.5;
}

__device__ float sRGBToLinear(int value) {
	float v = (float)value / 255;
	if (v <= 0.04045) return v / 12.92;
	else return powf((v + 0.055) / 1.055, 2.4);
}

float signPow(float value, float exp) {
	return copysignf(powf(fabsf(value), exp), value);
}

constexpr char characters[84] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";
