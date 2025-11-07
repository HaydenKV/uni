#ifndef DJIVIDEOCAPTION_H
#define DJIVIDEOCAPTION_H

#include <vector>
#include <filesystem>

struct DJIVideoCaption
{
    int frameNum;           // frame number
    double time;            // time at which the image was taken
    int iso;                // 
    double shutterHz;       // shutter frequency
    double fnum;            // [f stop value]
    double latitude;        // [degrees]
    double longitude;       // [degrees]
    double altitude;        // [m]
};

std::vector<DJIVideoCaption> getVideoCaptions(const std::filesystem::path & captionPath);

#endif
