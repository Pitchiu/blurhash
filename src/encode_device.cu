#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "common.h"

#include "encode_device.cuh"

void encodeDevice(const ProgramOptions& options)
{
    int xComponents = options.xComponents;
	int yComponents = options.yComponents;

	int acCount = xComponents * yComponents - 1;
	float* deviceAc;
	CHECK_CUDA_ERROR(cudaMalloc(&deviceAc, 3*acCount*sizeof(float)));

	
	char* encodedACValuesBuffer;
	CHECK_CUDA_ERROR(cudaMalloc(&encodedACValuesBuffer, sizeof(char)*2*acCount));

	for (auto const& filename : options.images)
	{
		std::string path(options.path + filename);
		const char* hash = encodeFile(xComponents, yComponents, path.c_str(), deviceAc, encodedACValuesBuffer);

		if (!hash)
		{
			fprintf(stderr, "Failed to load image file \"%s\".\n", path.c_str());
			exit(EXIT_FAILURE);
		}
		printf("%s\n", hash);
	}
	CHECK_CUDA_ERROR(cudaFree(deviceAc));
	CHECK_CUDA_ERROR(cudaFree(encodedACValuesBuffer));
}

const char* encodeFile(int xComponents, int yComponents, const char* filename, float* deviceAc, char* encodedACValuesBuffer)
{
	int width, height, channels;

	unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);
	if (!data) return NULL;

	const char* hash = encodePixels(data, xComponents, yComponents, width, height, deviceAc, encodedACValuesBuffer);
	stbi_image_free(data);
	return hash;
}


const char* encodePixels(uint8_t* data, int xComponents, int yComponents, int width, int height, float* deviceAc, char *encodedACValuesBuffer)
{
	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	float factors[yComponents * xComponents * 3];

	int N = width*height;
	uint8_t* pixels;
	CHECK_CUDA_ERROR(cudaMalloc(&pixels, N*3*sizeof(uint8_t)));
	CHECK_CUDA_ERROR(cudaMemcpy(pixels, data, N*3*sizeof(uint8_t), cudaMemcpyHostToDevice));

	float* vec_r;
	float* vec_g;
	float* vec_b;

	CHECK_CUDA_ERROR(cudaMalloc((void**)&vec_r, sizeof(float)*N));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&vec_g, sizeof(float)*N));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&vec_b, sizeof(float)*N));

	for (int y = 0; y < yComponents; y++) {
		for (int x = 0; x < xComponents; x++) {
			float factor[3];
			calculateFactor(pixels, x, y, width, height, factor, vec_r, vec_g, vec_b);
			factors[y * xComponents * 3 + x * 3 + 0] = factor[0];
			factors[y * xComponents * 3 + x * 3 + 1] = factor[1];
			factors[y * xComponents * 3 + x * 3 + 2] = factor[2];
		}
	}

	CHECK_CUDA_ERROR(cudaFree(vec_r));
	CHECK_CUDA_ERROR(cudaFree(vec_g));
	CHECK_CUDA_ERROR(cudaFree(vec_b));
	CHECK_CUDA_ERROR(cudaFree(pixels));


	float* dc = factors;
	float* ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char* ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encodeInt(sizeFlag, 1, ptr);
	float maximumValue;
	if (acCount > 0)
	{
		float actualMaximumValue = 0;
  		CHECK_CUDA_ERROR(cudaMemcpy(deviceAc, ac, 3*acCount*sizeof(float), cudaMemcpyHostToDevice));

		thrust::device_ptr<float> deviceAcPtr(deviceAc);
		actualMaximumValue = fabsf(*(thrust::max_element(deviceAcPtr, deviceAcPtr + 3*acCount, compare())));

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encodeInt(quantisedMaximumValue, 1, ptr);
	}
	else
	{
		maximumValue = 1;
		ptr = encodeInt(0, 1, ptr);
	}

	ptr = encodeInt(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	int numblocks = acCount / 128 + 1;
	encodeACValues<<<numblocks, 128>>>(deviceAc, encodedACValuesBuffer, maximumValue, acCount);
	cudaDeviceSynchronize();
  	CHECK_CUDA_ERROR(cudaMemcpy(buffer + 6, encodedACValuesBuffer, 2*acCount*sizeof(char), cudaMemcpyDeviceToHost));

	return buffer;
}


void calculateFactor(uint8_t* pixels, int xComponent, int yComponent, int width, int height, float result[3], float* vec_r, float* vec_g, float* vec_b)
{
	float r = 0, g = 0, b = 0;
	float normalisation = (xComponent == 0 && yComponent == 0) ? 1 : 2;
	int N = width * height;

	int numblocks = N/1024 + 1;
	calculatePixelColors<<<numblocks, 1024>>>(pixels, vec_r, vec_g, vec_b, xComponent, yComponent, width, height);
	cudaDeviceSynchronize();

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