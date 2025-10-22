#ifndef CALIBRATE_H
#define CALIBRATE_H

#include <filesystem>

void calibrateCamera(const std::filesystem::path & configPath,
                    const std::filesystem::path& outputDirectory = {});

#endif