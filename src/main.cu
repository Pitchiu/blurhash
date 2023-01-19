#include <vector>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "common.h"
#include "timer.h"


const char* blurHashForPixels(int xComponents, int yComponents, int width, int height, uint8_t* pixels, size_t bytesPerRow);
const char* blurHashForFile(int xComponents, int yComponents, const char* filename);
char* encode_int(int value, int length, char* destination);
void multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t* pixels, float result[3], float* vec_r,float* vec_g,float* vec_b );
int encodeAC(float r, float g, float b, float maximumValue);
int encodeDC(float r, float g, float b);

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void checkLast(const char* const file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


int main(int argc, char **argv) 
{
	if(argc != 4)
	{
		fprintf(stderr, "Usage: %s x_components y_components imagefile\n", argv[0]);
		return 1;
	}

	int xComponents = atoi(argv[1]);
	int yComponents = atoi(argv[2]);
	char* filename = argv[3];

	Stopwatch<> timer;
	const char* hash = blurHashForFile(xComponents, yComponents, filename);
	for(int i=0; i<99; i++)
	{
		const char* br = blurHashForFile(xComponents, yComponents, filename);

	}

	if (!hash)
	{
		fprintf(stderr, "Failed to load image file \"%s\".\n", filename);
		return 1;
	}
	printf("%s\n", hash);

	auto time = timer.elapsed_time<unsigned int, std::chrono::milliseconds>();
	printf("Time elapsed: %dms\n", time);

	return 0;
}

const char* blurHashForFile(int xComponents, int yComponents, const char* filename)
{
	int width, height, channels;

	unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);
	if (!data) return NULL;

	const char* hash = blurHashForPixels(xComponents, yComponents, width, height, data, width * 3);
	stbi_image_free(data);
	return hash;
}


const char* blurHashForPixels(int xComponents, int yComponents, int width, int height, uint8_t* data, size_t bytesPerRow)
{
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	float* factors = new float[yComponents * xComponents * 3];

	int N = width*height;
	uint8_t* pixels;
	CHECK_CUDA_ERROR(cudaMalloc(&pixels, N*3*sizeof(uint8_t)));
	CHECK_CUDA_ERROR(cudaMemcpy(pixels, data, N*3*sizeof(uint8_t), cudaMemcpyHostToDevice));

	float* vec_r;
	float* vec_g;
	float* vec_b;

	cudaMalloc((void**)&vec_r, sizeof(float)*N);
	cudaMalloc((void**)&vec_g, sizeof(float)*N);
	cudaMalloc((void**)&vec_b, sizeof(float)*N);

	for (int y = 0; y < yComponents; y++) {
		for (int x = 0; x < xComponents; x++) {
			float factor[3];
			multiplyBasisFunction(x, y, width, height, pixels, factor, vec_r, vec_g, vec_b);
			factors[y * xComponents * 3 + x * 3 + 0] = factor[0];
			factors[y * xComponents * 3 + x * 3 + 1] = factor[1];
			factors[y * xComponents * 3 + x * 3 + 2] = factor[2];
		}
	}

	cudaFree(vec_r);
	cudaFree(vec_g);
	cudaFree(vec_b);
	CHECK_CUDA_ERROR(cudaFree(pixels));


	float* dc = factors;
	float* ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char* ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	if (acCount > 0) {
		float actualMaximumValue = 0;
		for (int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	}
	else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for (int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;
	delete[] factors;

	return buffer;
}

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

void multiplyBasisFunction(int xComponent, int yComponent, int width, int height, uint8_t* pixels, float result[3],  float* vec_r,float* vec_g,float* vec_b)
{
	float r = 0, g = 0, b = 0;
	float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;
	int N = width * height;

	int numblocks = N/1024 + 1;
	calculatePixelColors<<<numblocks, 1024>>>(pixels, vec_r, vec_g, vec_b, xComponent, yComponent, width, height);
	cudaDeviceSynchronize();
	CHECK_LAST_CUDA_ERROR();

	thrust::device_ptr<float> r_ptr(vec_r);
	thrust::device_ptr<float> g_ptr(vec_g);
	thrust::device_ptr<float> b_ptr(vec_b);

	r = thrust::reduce(r_ptr, r_ptr + N, (float) 0);
	g = thrust::reduce(g_ptr, g_ptr + N, (float) 0);
	b = thrust::reduce(b_ptr, b_ptr + N, (float) 0);

	float scale = normalisation / (width * height);

	result[0] = r * scale;
	result[1] = g * scale;
	result[2] = b * scale;
}

int encodeDC(float r, float g, float b) {
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

int encodeAC(float r, float g, float b, float maximumValue) {
	int quantR = fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5) * 9 + 9.5)));
	int quantG = fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5) * 9 + 9.5)));
	int quantB = fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5) * 9 + 9.5)));

	return quantR * 19 * 19 + quantG * 19 + quantB;
}

char* encode_int(int value, int length, char* destination) {
	int divisor = 1;
	for (int i = 0; i < length - 1; i++) divisor *= 83;

	for (int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}
