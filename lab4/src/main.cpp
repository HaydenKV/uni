#include <cstdlib>
#include <filesystem>
#include <print>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Camera.h"

int main(int argc, char* argv[])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this help message}"
        "{@config         |  <none>  | path to configuration XML}"
        "{calibrate c     |          | perform camera calibration for given configuration XML}"
        "{export e        |          | export files}"
        "{verbose v       |  0       | verbosity level: 0: none, 1: draw corners, 2: draw boxes}"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 4");
    
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool hasCalibrate = parser.has("calibrate");
    bool hasExport = parser.has("export");
    std::filesystem::path configPath = parser.get<std::string>("@config");
    int verbosity = parser.get<int>("verbose");

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
        // Load calibration
        if (!std::filesystem::exists(cameraPath))
        {
            std::println("File: {} does not exist", cameraPath.string());
            return EXIT_FAILURE;
        }
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        assert(fs.isOpened());
        fs["camera"] >> cam;

        // Display loaded calibration data
        cam.printCalibration();

        // Reconstruct extrinsic parameters (camera pose) for each chessboard image
        chessboardData.recoverPoses(cam);
    }

    // Visualisation
    if (verbosity > 0)
    {
        if (verbosity == 1)
        {
            chessboardData.drawCorners();
        }
        else if (verbosity == 2)
        {
            chessboardData.drawBoxes(cam);
        }

        for (const auto & chessboardImage : chessboardData.chessboardImages)
        {
            if (hasExport)
            {
                std::filesystem::path outputPath = outputDirectory / chessboardImage.filename;
                cv::imwrite(outputPath.string(), chessboardImage.image);
            }
            else
            {
                cv::imshow("Calibration images (press ESC to quit, any other key to continue)", chessboardImage.image);
                char c = static_cast<char>(cv::waitKey(0));
                if (c == 27) // ESC to quit, any other key to continue
                    break;
            }
        }
    }

    return EXIT_SUCCESS;
}

