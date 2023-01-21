#include "parser.cuh"

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

bool listImages(const char* directory, ProgramOptions& options)
{
    std::string directoryStr(directory);
    if(!boost::algorithm::ends_with(directoryStr, "/"))
        directoryStr += "/";
    options.path = directoryStr;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (directory)) != NULL)
    {
    while ((ent = readdir (dir)) != NULL)
        {
            if(boost::algorithm::ends_with(ent->d_name, ".jpg"))
            {
                options.images.push_back(ent->d_name);
            }
        }
        closedir (dir);
        return true;
    }
    else
    {
        printf("Error while opening directory\n");
        return false;
    }
    return true;
}

bool parseDir(int argc, char **argv, ProgramOptions& options)
{
    char *directory;
    if(cmdOptionExists(argv, argv+argc, "-d"))
    {
        directory = getCmdOption(argv, argv + argc, "-d");
        if(directory)
        {
            if(!listImages(directory, options))
                return false;
        }
        else
        {
            printf("Image directory is missing\n");
            return false;
        }
    }
    else
    {
        printf("Directory (-d) required");
        return false;
    }
    return true;
}

bool parseMode(int argc, char **argv, ProgramOptions& options)
{
    if(cmdOptionExists(argv, argv+argc, "-m"))
    {
        char *modeString = getCmdOption(argv, argv + argc, "-m");
        if(!modeString)
        {
            printf("Mode is missing\n");
            return false;
        }

        if(strcmp(modeString, "gpu") == 0)
            options.mode = gpuOnly;
        else if(strcmp(modeString, "cpu") == 0)
            options.mode = cpuOnly;
        else if(strcmp(modeString, "comparison") == 0)
            options.mode = comparison;
        else
        {
            printf("Mode not recognized\n");
            return false;
        }
    }
    else
    {
        printf("Mode (-m) is missing\n");
        return false;
    }
    return true;
}

bool parseComponents(int argc, char **argv, ProgramOptions& options)
{
    if(!cmdOptionExists(argv, argv+argc, "-x") || !cmdOptionExists(argv, argv+argc, "-y"))
    {
        printf("Components are missing (-x & -y)\n");
        return false;
    }
    char *componentString;
    componentString = getCmdOption(argv, argv + argc, "-x");
    if(!componentString || atoi(componentString)<0 || atoi(componentString)>100)
    {
        printf("-x component is incorrect\n");
        return false;
    }
    options.xComponents = atoi(componentString);    
    
    componentString = getCmdOption(argv, argv + argc, "-y");
    if(!componentString || atoi(componentString)<0 || atoi(componentString)>100)
    {
        printf("-y component is incorrect\n");
        return false;
    }
    options.yComponents = atoi(componentString);

    return true;
}

bool parseInput(int argc, char **argv, ProgramOptions &options)
{
    if(cmdOptionExists(argv, argv+argc, "-h") || argc==1)
    {
        printf("Usage: blurhash -d {directory_path} -m [gpu|cpu|comparison] -x {components_x} -y {components_y} \n");
        return false;
    }

    if(!parseDir(argc, argv, options))
        return false;

    if(!parseMode(argc, argv, options))
        return false;

    if(!parseComponents(argc, argv, options))
        return false;

    return true;
}