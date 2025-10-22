#include <cstdlib>
#include <filesystem>
#include <print>
#include "Camera.h"
#include "confidence_region_demo.h"

int main(int argc, char* argv[])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |        | print this message}"
        "{@config         | <none> | path to configuration XML}"
        "{calibrate c     |        | perform camera calibration for given configuration XML}"
        "{export e        |        | export files to the ./out/ directory}"
        "{interactive i   | 2      | interactivity (0:none, 1:last image, 2:all images)}"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 8");
    
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool hasCalibrate = parser.has("calibrate");
    bool hasExport = parser.has("export");
    std::filesystem::path configPath = parser.get<std::string>("@config");
    int interactive = parser.get<int>("interactive");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    // Prepare output directory
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
        std::println("Output directory set to {}", outputDirectory.string());
    }

    // Check if configuration exists
    if (!std::filesystem::exists(configPath))
    {
        std::println("File: {} does not exist", configPath.string());
        return EXIT_FAILURE;
    }

    // Read chessboard data using configuration file
    ChessboardData chessboardData(configPath);

    // Camera calibration file
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";

    Camera cam;
    if (hasCalibrate)
    {
        // Calibrate camera from chessboard data
        cam.calibrate(chessboardData);

        // Write camera calibration to file
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
        fs << "camera" << cam;
        fs.release();
    }
    else
    {
        // Do confidence region demo

        // Read camera calibration using default camera file path
        if (!std::filesystem::exists(cameraPath))
        {
            std::println("File: {} does not exist", cameraPath.string());
            return EXIT_FAILURE;
        }
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        assert(fs.isOpened());
        fs["camera"] >> cam;

        // Run confidence region demo
        confidenceRegionDemo(cam, chessboardData, outputDirectory, interactive);
    }

    return EXIT_SUCCESS;
}
