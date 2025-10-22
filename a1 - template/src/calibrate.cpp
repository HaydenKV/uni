#include <filesystem>
#include "calibrate.h"

void calibrateCamera(const std::filesystem::path & configPath)
{
    // TODO
    // - Read XML at configPath
    // - Parse XML and extract relevant frames from source video containing the chessboard
    // - Perform camera calibration
    // - Write the camera matrix and lens distortion parameters to camera.xml file in same directory as configPath
    // - Visualise the camera calibration results
}
