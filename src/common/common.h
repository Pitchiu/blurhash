#ifndef __BLURHASH_COMMON_H__
#define __BLURHASH_COMMON_H__
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr char characters[84] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

static int linearTosRGB(float value)
{
	float v = fmaxf(0, fminf(1, value));
	if (v <= 0.0031308) return v * 12.92 * 255 - 0.5;
	else return (1.055 * powf(v, 1 / 2.4) - 0.055) * 255 - 0.5;
}

static int encodeDC(float r, float g, float b)
{
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

static char* encodeInt(int value, int length, char* destination)
{
	int divisor = 1;
	for (int i = 0; i < length - 1; i++) divisor *= 83;

	for (int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}

#endif