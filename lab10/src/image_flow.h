#ifndef IMAGE_FLOW_H
#define IMAGE_FLOW_H

#include <filesystem>

void runImageFlow(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory);

#endif