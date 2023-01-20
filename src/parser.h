#include <string>
#include <algorithm>
#include <list>

enum Mode
{
	gpuOnly,
	cpuOnly,
	comparison
};

struct ProgramOptions
{
	std::string path;
	std::list<std::string> images;
    Mode mode;
	int xComponents;
	int yComponents;
};

bool parseInput(int argc, char **argv, ProgramOptions &programOptions);


