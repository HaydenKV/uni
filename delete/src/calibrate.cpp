#include "calibrate.h"
#include "Camera.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <filesystem>
#include <format>

void calibrateCamera(const std::filesystem::path & configPath, const std::filesystem::path& outputDirectory)
{
    // - Read XML at configPath
    // - Parse XML and extract relevant frames from source video containing the chessboard
    // - Perform camera calibration
    // - Write the camera matrix and lens distortion parameters to camera.xml file in same directory as configPath
    // - Visualise the camera calibration results

    std::cout << "Loading calibration configuration from: " << configPath.string() << std::endl;
    
    // Check if configuration exists
    if (!std::filesystem::exists(configPath))
    {
        std::cerr << "Configuration file does not exist: " << configPath.string() << std::endl;
        return;
    }
    
    // Load chessboard data from configuration (this handles video parsing)
    ChessboardData chessboardData(configPath);
    
    if (chessboardData.chessboardImages.empty())
    {
        std::cerr << "No chessboard images found for calibration" << std::endl;
        return;
    }
    
    std::cout << "Found " << chessboardData.chessboardImages.size() 
              << " images with detected chessboards" << std::endl;
    
    // Create camera object and calibrate
    Camera camera;
    camera.calibrate(chessboardData);
    
    // Write calibration to camera.xml in same directory as config
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";
    std::cout << "Writing camera calibration to: " << cameraPath.string() << std::endl;
    
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open file for writing: " << cameraPath.string() << std::endl;
        return;
    }
    
    fs << "camera" << camera;
    fs.release();
    
    std::cout << "Camera calibration completed successfully!" << std::endl;
    
    // Optional: Display validation visualization
    // Show some calibration images with detected corners
    chessboardData.drawCorners();
    chessboardData.drawBoxes(camera);
    

    bool shouldExport = !outputDirectory.empty();
    
    if (shouldExport)
    {
        std::cout << "Exporting validation images to: " << outputDirectory.string() << std::endl;
        
        // Export all calibration images with validation overlays
        for (size_t i = 0; i < chessboardData.chessboardImages.size(); ++i)
        {
            const auto& img = chessboardData.chessboardImages[i];
            
            // Create filename for export
            std::string exportName = std::format("calibration_validation_{:03d}.png", i);
            std::filesystem::path exportPath = outputDirectory / exportName;
            
            // Save the image with overlays
            cv::imwrite(exportPath.string(), img.image);
            
            if (i == 0) // Print confirmation for first image
            {
                std::cout << "  Saved: " << exportName << std::endl;
            }
        }
        
        std::cout << "  ... (exported " << chessboardData.chessboardImages.size() 
                  << " validation images total)" << std::endl;
    }
    
    // Interactive validation display
    std::cout << "\nValidation Display:" << std::endl;
    std::cout << "- Blue/Green dots: Detected chessboard corners" << std::endl;
    std::cout << "- Blue base: Chessboard plane" << std::endl;
    std::cout << "- Red edges: 3D box rising from board" << std::endl;
    std::cout << "- Green lid: Top of 3D validation box" << std::endl;
    
    if (shouldExport)
    {
        std::cout << "\nValidation images exported. No interactive display when using --export flag." << std::endl;
        std::cout << "(All images have been exported to " << outputDirectory.string() << ")" << std::endl;
    }
    else
    {
        std::cout << "\nPress any key in image window to browse through all validation images..." << std::endl;
        
        // Show all images when not exporting (same as exporting, just no file save)
        for (size_t i = 0; i < chessboardData.chessboardImages.size(); ++i)
        {
            const auto& img = chessboardData.chessboardImages[i];
            std::string windowTitle = std::format("Calibration Validation ({}/{}): {}", 
                                                 i + 1, chessboardData.chessboardImages.size(),
                                                 img.filename.string());
            
            cv::imshow(windowTitle, img.image);
            int key = cv::waitKey(0);
            
            // Allow ESC to exit early
            if (key == 27) // ESC key
            {
                std::cout << "Validation display interrupted by user" << std::endl;
                break;
            }
        }
    }
    
    cv::destroyAllWindows();
    
    std::cout << "\nCalibration validation complete!" << std::endl;
}
