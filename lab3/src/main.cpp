#include <cstdlib>
#include <iostream>
#include <string>
#include <print>
#include <filesystem>
#include <format>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "BufferedVideo.h"
#include "DuckDetectorONNX.h"


int main(int argc, char *argv[])
{
    cv::String keys = 
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this message}"
        "{@input          | <none>   | input can be a path to a video (e.g., ../data/duck.MOV)}"
        "{export e        |          | export output files to the ./out/ directory}"
        "{annotate        |          | annotate output with centroid/area info}" //added this line
    ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 3");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool doExport = parser.has("export");
    bool doAnnotate = parser.has("annotate");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (!std::filesystem::exists(inputPath))
    {
        std::println("File: {} does not exist", inputPath.string());
        return EXIT_FAILURE;
    }

    // Prepare output directory
    std::filesystem::path outputDirectory;
    std::filesystem::path outputImageDirectory;
    if (doExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::println("Creating directory {}", outputDirectory.string());
            std::filesystem::create_directory(outputDirectory);
        }

        outputImageDirectory = outputDirectory / "images";
        if (!std::filesystem::exists(outputImageDirectory))
        {
            std::println("Creating directory {}", outputImageDirectory.string());
            std::filesystem::create_directory(outputImageDirectory);
        }
    }

    // Run the detector on the video

    // ONNX Runtime
    std::filesystem::path onnx_file = "../data/duck_with_postprocessing.onnx";
    DuckDetectorONNX detector(onnx_file.string());

    // Prepare output file path
    std::filesystem::path outputPath;
    if (doExport)
    {
        std::string outputFilename = inputPath.stem().string()
                                + "_"
                                + "eval"
                                + inputPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    cv::VideoCapture cap(inputPath.string());
    if (cap.isOpened())
    {
        // Get frame rate from source video
        double fps = cap.get(cv::CAP_PROP_FPS);

        // Initialize buffered video reader
        BufferedVideoReader bufferedVideoReader(5);
        bufferedVideoReader.start(cap);

        // Initialize buffered video writer
        cv::VideoWriter video;
        BufferedVideoWriter bufferedVideoWriter(3);
        bool isOpened = false;
        
        cv::Mat img;
        int idxFrame = 0;
        while (true)
        {
            // Get next input frame using buffered reader
            img = bufferedVideoReader.read();
            if (img.empty())
            {
                break;
            }
            
            std::string baseName = inputPath.stem().string();
            std::string frameName = std::format("{}_{:05d}", baseName, idxFrame);
            std::string frameFilename = frameName + ".png";
            std::string frameFilenameEval =  frameName + "_eval.png";
            
            // Run duck detector
            std::print("Processing {}...", frameName);
            cv::Mat imgout = detector.detect(img);
            std::println(" done");
            
            if (doExport && !isOpened)
            {
                // Set output video codec
                static const int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                video.open(outputPath.string(), codec, fps, imgout.size());
                bufferedVideoWriter.start(video);
                isOpened = true;
            }

            if (doExport)
            {
                if (idxFrame % 15 == 0)
                {
                    std::filesystem::path framePath = outputImageDirectory / frameFilename;
                    cv::imwrite(framePath.string(), img);
                    std::filesystem::path frameAnnotatedPath = outputImageDirectory / frameFilenameEval;
                    cv::imwrite(frameAnnotatedPath.string(), imgout);
                }

                bufferedVideoWriter.write(imgout);
            }
            else
            {
                cv::String title = "Eval. Press ESC to quit.";
                cv::imshow(title, imgout);
                // int waitTime = std::max(1, static_cast<int>(1000.0/fps)); // Run continuously
                int waitTime = 0; // Pause on each frame
                char c = cv::waitKey(waitTime);
                if (c == 27)
                {
                    break;
                }
            }

            idxFrame++;
        }

        if (doExport && isOpened)
        {
            bufferedVideoWriter.stop();
        }
        bufferedVideoReader.stop();
    }
    return EXIT_SUCCESS;
}


