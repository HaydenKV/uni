#include <cstdlib>
#include <cassert>
#include <string>  
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include "calibrate.h"
#include "visualNavigation.h"

int main(int argc, char* argv [])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this help message}"
        "{@input          | <none>   | path to input video or configuration XML}"
        "{calibrate c     |          | perform camera calibration for given configuration XML}"
        "{scenario s      | 3        | run visual navigation on input video with scenario type (1:tag, 2:duck, 3:point)}"
        "{interactive i   | 0        | interactivity (0:none, 1:last frame, 2:all frames)}"
        "{export e        |          | export video}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Assignment 1");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    int scenario = parser.get<int>("scenario");
    int interactive = parser.get<int>("interactive");
    bool hasExport = parser.has("export");
    bool hasCalibrate = parser.has("calibrate");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    std::filesystem::path outputDirectory;
    if (hasExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::cout << "Creating directory " << outputDirectory.string() << std::endl;
            std::filesystem::create_directory(outputDirectory);
        }
    }

    if (hasCalibrate)
    {
        std::cout << "Calibrating camera" << std::endl;
        std::cout << "Configuration file: " << inputPath.string() << std::endl;
        calibrateCamera(inputPath);
    }
    else
    {
        assert(1 <= scenario && scenario <= 3);
        assert(0 <= interactive && interactive <= 2);
        std::cout << "Running visual navigation" << std::endl;
        std::cout << "Input video: " << inputPath.string() << std::endl;
        std::filesystem::path cameraPath = inputPath.parent_path() / "camera.xml"; 
        runVisualNavigationFromVideo(inputPath, cameraPath, scenario, interactive, outputDirectory);
    }

    return EXIT_SUCCESS;
}
