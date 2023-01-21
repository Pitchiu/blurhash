#include <iostream>

#include "common/timer.h"
#include "gpu/encode_device.cuh"
#include "cpu/encode_cpu.cuh"

int main(int argc, char **argv) 
{
	ProgramOptions options;
	if(!parseInput(argc, argv, options))
		return -1;

	// GPU version
	if(options.mode != cpuOnly)
	{
		Stopwatch<> timer;
		encodeDevice(options);
		auto time = timer.elapsed_time<unsigned int, std::chrono::milliseconds>();
		printf("GPU: Total time elapsed: %dms\n", time);
	}

	// CPU version
	if(options.mode != gpuOnly)
	{
		Stopwatch<> timer;
		encodeCPU(options);
		auto time = timer.elapsed_time<unsigned int, std::chrono::milliseconds>();
		printf("CPU: Total time elapsed: %dms\n", time);
	}

	return 0;
}