#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include <filesystem>

void runVisualOdometryFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory = "");

#endif
