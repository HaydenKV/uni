#include <cstdlib>
#include <cassert>
#include <string>  
#include <filesystem>
#include <print>
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
        "{scenario s      | 6        | run visual navigation on input video with scenario type (4:flight, 5:tag, 6:room)}"
        "{interactive i   | 0        | interactivity (0:none, 1:last frame, 2:all frames)}"
        "{export e        |          | export video}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Assignment 2");

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
            std::println("Creating directory {}", outputDirectory.string());
            std::filesystem::create_directory(outputDirectory);
        }
    }

    if (hasCalibrate)
    {
        std::println("Calibrating camera");
        std::println("Configuration file: {}", inputPath.string());
        calibrateCamera(inputPath, hasExport ? outputDirectory : std::filesystem::path{});
    }
    else
    {
        assert(1 <= scenario && scenario <= 6); // Optionally support earlier scenarios from Assignment 1
        assert(0 <= interactive && interactive <= 2);
        std::println("Running visual navigation");
        std::println("Input video: {}", inputPath.string());
        std::filesystem::path cameraPath = inputPath.parent_path() / "camera.xml"; 
        runVisualNavigationFromVideo(inputPath, cameraPath, scenario, interactive, outputDirectory);
    }

    return EXIT_SUCCESS;
}

