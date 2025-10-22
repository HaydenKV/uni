#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include "DJIVideoCaption.h"

std::vector<DJIVideoCaption> getVideoCaptions(const std::filesystem::path & captionPath)
{
    assert(std::filesystem::exists(captionPath));

    std::vector<DJIVideoCaption> caps;
    
    FILE* file = std::fopen(captionPath.string().c_str(), "r");
    if (!file) {
        return caps;
    }
    
    int frameNum;
    int h1, m1, s1, ms1, h2, m2, s2, ms2;
    char buffer[2048];
    
    // Read each subtitle entry
    while (std::fscanf(file, "%d\n", &frameNum) == 1) {
        // Read timestamp
        if (std::fscanf(file, "%d:%d:%d,%d --> %d:%d:%d,%d\n",
                       &h1, &m1, &s1, &ms1, &h2, &m2, &s2, &ms2) != 8) {
            break;
        }
        
        // Skip font tag line
        if (!std::fgets(buffer, sizeof(buffer), file)) break;
        
        // Skip date/time line  
        if (!std::fgets(buffer, sizeof(buffer), file)) break;
        
        // Read data line with sensor/GPS info
        if (!std::fgets(buffer, sizeof(buffer), file)) break;
        
        // Parse data fields
        int iso = 0;
        float shutterHz = 0.0f;
        int fnumInt = 0;
        double latitude = 0.0, longitude = 0.0, altitude = 0.0;
        
        char* ptr;
        if ((ptr = std::strstr(buffer, "[iso : ")) != nullptr) {
            std::sscanf(ptr, "[iso : %d]", &iso);
        }
        
        if ((ptr = std::strstr(buffer, "[shutter : 1/")) != nullptr) {
            std::sscanf(ptr, "[shutter : 1/%f]", &shutterHz);
        }
        
        if ((ptr = std::strstr(buffer, "[fnum : ")) != nullptr) {
            std::sscanf(ptr, "[fnum : %d]", &fnumInt);
        }
        
        if ((ptr = std::strstr(buffer, "[latitude : ")) != nullptr) {
            std::sscanf(ptr, "[latitude : %lf]", &latitude);
        }
        
        // Note: DJI misspells "longitude" as "longtitude" in their SRT files
        if ((ptr = std::strstr(buffer, "[longtitude : ")) != nullptr) {
            std::sscanf(ptr, "[longtitude : %lf]", &longitude);
        }
        
        if ((ptr = std::strstr(buffer, "[altitude: ")) != nullptr) {
            std::sscanf(ptr, "[altitude: %lf]", &altitude);
        }
        
        // Create caption entry
        DJIVideoCaption caption;
        caption.frameNum = frameNum;
        caption.time = h1 * 3600.0 + m1 * 60.0 + s1 + ms1 / 1000.0;
        caption.iso = iso;
        caption.shutterHz = shutterHz;
        caption.fnum = fnumInt / 100.0;
        caption.latitude = latitude;
        caption.longitude = longitude;
        caption.altitude = altitude;
        
        caps.push_back(caption);
        
        // Skip remaining lines until blank line
        while (std::fgets(buffer, sizeof(buffer), file)) {
            if (buffer[0] == '\n' || buffer[0] == '\r') {
                break;
            }
        }
    }
    
    std::fclose(file);
    return caps;
}