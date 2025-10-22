#ifndef CONFIDENCE_REGION_DEMO_H
#define CONFIDENCE_REGION_DEMO_H

#include <filesystem>
#include "Camera.h"

void confidenceRegionDemo(const Camera & cam, ChessboardData & chessboardData, const std::filesystem::path & outputDirectory, int interactive);

#endif