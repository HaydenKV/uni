#ifndef VISUALNAVIGATION_H
#define VISUALNAVIGATION_H

#include <filesystem>

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario = 3, int interactive = 0, const std::filesystem::path & outputDirectory = "");

#endif