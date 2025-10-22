#include <filesystem>
#include <format>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "Camera.h"
#include "calibrate.h"

void calibrateCamera(const std::filesystem::path & configPath,
                     const std::filesystem::path& outputDirectory)
{
    // Read chessboard data using configuration file
    ChessboardData chessboardData(configPath);

    // Calibrate camera from chessboard data
    Camera cam;
    cam.calibrate(chessboardData);

    // Write camera calibration to file
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
    fs << "camera" << cam;
    fs.release();

    // Draw overlays we want to export / show
    chessboardData.drawBoxes(cam);

    const bool shouldExport = !outputDirectory.empty();

    if (shouldExport)
    {
        std::filesystem::create_directories(outputDirectory);

        int idx = 0;
        for (const auto& img : chessboardData.chessboardImages)
        {
            const std::string name = std::format("calibration_{:06d}.jpg", idx++);
            cv::imwrite((outputDirectory / name).string(), img.image);
        }

        // No viewer when exporting (matches previous labs’ behavior)
        return;
    }
    
    // Visualise the camera calibration results
    // chessboardData.drawBoxes(cam);
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        cv::imshow("Calibration images", chessboardImage.image);
        char c = static_cast<char>(cv::waitKey(0));
        if (c == 27 || c == 'q' || c == 'Q') // ESC/q/Q to quit
            break;
    }

    cv::destroyAllWindows();
}
