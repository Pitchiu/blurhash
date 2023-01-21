#ifndef __BLURHASH_PARSER_H__
#define __BLURHASH_PARSER_H__

#include <string>
#include <algorithm>
#include <list>
#include <iostream>
#include <string>
#include <dirent.h>
#include <boost/algorithm/string/predicate.hpp>

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

#endif