#include <filesystem>
#include <string>
#include <print>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include "BufferedVideo.h"
#include "Camera.h"
#include "image_flow.h"

void runImageFlow(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();

    // TODO: Lab 10
    int divisor         = 1;                    // Image scaling factor
    int imgModulus      = 1;                    // Take frames divisible by this number
    int maxNumFeatures  = 0;                    // Maximum number of features per frame
    int minNumFeatures  = 0;                    // Minimum number of feature per frame

    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_"
                                   + std::to_string(divisor)
                                   + "_"
                                   + std::to_string(imgModulus)
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
    Camera camera;
    assert(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> camera;

    // Display loaded calibration data
    camera.printCalibration();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);

    std::println("Input video: {}", videoPath.string());
    std::println("Total number of frames: {}", nFrames);
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::println("Input video frame rate: {}", fps);
    std::println("Input video dimensions: [{} x {}]",
                cap.get(cv::CAP_PROP_FRAME_WIDTH),
                cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::println("{:>20}: {}", "maxNumFeatures", maxNumFeatures);
    std::println("{:>20}: {}", "minNumFeatures", minNumFeatures);
    std::println("{:>20}: {}", "divisor", divisor);
    std::println("{:>20}: {}", "imgModulus", imgModulus);
    std::println("");

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        double outputFps    = fps/imgModulus;
        cv::Size frameSize;
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Image flow
    // TODO: Lab 10

    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();
}
