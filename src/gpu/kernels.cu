#include "kernels.cuh"

__global__ void calculatePixelColors(uint8_t* pixels, float* vec_r, float* vec_g, float* vec_b, int xComponent, int yComponent, int width, int height)
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	int n = width*height;

	if(globalId < n)
	{
		int row = globalId / width;	//row
		int col = globalId - row*width;	//col
		float basis = cosf(M_PI * xComponent * col / width) * cosf(M_PI * yComponent * row / height);

		vec_r[globalId] = basis * sRGBToLinear(pixels[3*globalId]);
		vec_g[globalId] = basis * sRGBToLinear(pixels[3*globalId + 1]);
		vec_b[globalId] = basis * sRGBToLinear(pixels[3*globalId + 2]);
	}
}

__global__ void encodeACValues(float *acBufferDevice, char* encodedACValuesBuffer, float maximumValue, int n)
{
 	int globalId = blockIdx.x * blockDim.x + threadIdx.x;

	if(globalId < n)
	{
		float r = acBufferDevice[globalId*3 + 0];
		float g = acBufferDevice[globalId*3 + 1];
		float b = acBufferDevice[globalId*3 + 2];
		int quantR = fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5) * 9 + 9.5)));
		int quantG = fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5) * 9 + 9.5)));
		int quantB = fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5) * 9 + 9.5)));
		int result = quantR * 19 * 19 + quantG * 19 + quantB;
		encodeIntDevice(encodedACValuesBuffer, globalId*2, result);
	}

}

__device__ void encodeIntDevice(char *buffer, int index, int number)
{
	buffer[index] = charactersDevice[number / 83];
	buffer[index+1] = charactersDevice[number % 83];
}

__device__ float sRGBToLinear(uint8_t value)
{
	float v = (float)value / 255;
	if (v <= 0.04045) return v / 12.92;
	else return powf((v + 0.055) / 1.055, 2.4);
}

__device__ float signPow(float value, float exp)
{
	return copysignf(powf(fabsf(value), exp), value);
}