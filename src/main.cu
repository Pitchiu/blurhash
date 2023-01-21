#include <iostream>

#include "timer.h"
#include "encode_device.cuh"

int main(int argc, char **argv) 
{
	ProgramOptions options;
	if(!parseInput(argc, argv, options))
		return -1;

	Stopwatch<> timer;

	if(options.mode == gpuOnly)
	{
		encodeDevice(options);
	}
	else if(options.mode == cpuOnly)
	{
		//encodeCpu(options);
	}
	else if(options.mode == comparison)
	{

	}

	auto time = timer.elapsed_time<unsigned int, std::chrono::milliseconds>();
	printf("Time elapsed: %dms\n", time);

	return 0;
}