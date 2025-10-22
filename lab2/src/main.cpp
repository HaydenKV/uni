#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <print>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <nanobench.h>
#include "imagefeatures.h"

int main(int argc, char *argv[])
{
    cv::String keys = 
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this message}"
        "{@input          | <none>   | input can be a path to an image or video (e.g., ../data/lab.jpg)}"
        "{export e        |          | export output file to the ./out/ directory}"
        "{N               | 10       | maximum number of features to find}"
        "{detector d      | fast     | feature detector to use (e.g., harris, shi, aruco, fast)}"
        "{benchmark b     |          | run benchmark for all detectors}"
    ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 2");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool doExport = parser.has("export");
    int maxNumFeatures = parser.get<int>("N");
    bool doBenchmark = parser.has("benchmark");
    cv::String detector = parser.get<std::string>("detector");
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
        std::println("Output directory set to {}", outputDirectory.string());
    }

    // Prepare output file path
    std::filesystem::path outputPath;
    if (doExport)
    {
        std::string outputFilename = inputPath.stem().string()
                                   + "_"
                                   + detector
                                   + inputPath.extension().string();
        outputPath = outputDirectory / outputFilename;
        std::println("Output name: {}", outputPath.string());
    }

    // Check if input is an image or video (or neither)
    // bool isVideo = true; // TODO
    // bool isImage = true; // TODO
    
    // Try to read the input as an image
    cv::Mat inputImage = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
    bool isImage = !inputImage.empty();  // It’s an image if this isn’t empty

    // If not an image, try opening as a video
    cv::VideoCapture inputVideo;
    bool isVideo = false;
    if (!isImage) {
        inputVideo.open(inputPath.string());
        isVideo = inputVideo.isOpened();
    }

    if (!isImage && !isVideo)
    {
        std::println("Could not read file: {}", inputPath.string());
        return EXIT_FAILURE;
    }

    if (doBenchmark)
    {
        if (!isImage)
        {
            std::println("Benchmark can only be run on images, not videos.");
            return EXIT_FAILURE;
        }

        // Suppress console output during benchmark
        std::FILE* old_stdout = stdout;
        stdout = std::fopen("/dev/null", "w");

        // Create benchmark object
        ankerl::nanobench::Bench bench;
        
        // Capture benchmark output in a separate stringstream
        std::stringstream bench_output;
        bench.output(&bench_output);

        // TODO: Run the benchmarks for the 4 feature detectors
        bench
            .title("Feature Detector")
            .unit("op")
            .warmup(5)
            .minEpochIterations(122)
            .relative(true)
            .performanceCounters(false);

        bench.run("Harris", [&] {
            detectAndDrawHarris(inputImage, maxNumFeatures);
        });

        bench.run("Shi-Tomasi", [&] {
            detectAndDrawShiAndTomasi(inputImage, maxNumFeatures);
        });

        bench.run("FAST", [&] {
            detectAndDrawFAST(inputImage, maxNumFeatures);
        });

        bench.run("ArUco", [&] {
            detectAndDrawArUco(inputImage, maxNumFeatures);
        });

        // Restore console output
        std::fclose(stdout);
        stdout = old_stdout;

        // Print benchmark results
        std::println("\nBenchmark results:");
        std::print("{}", bench_output.str());

        return EXIT_SUCCESS;
    }

    if (isImage)
    {
        // Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
        cv::Mat output;
        if (detector == "harris"){
            output = detectAndDrawHarris(inputImage, maxNumFeatures);
        } else if (detector == "shi") {
            output = detectAndDrawShiAndTomasi(inputImage, maxNumFeatures);
        } else if (detector == "fast") {
            output = detectAndDrawFAST(inputImage, maxNumFeatures);
        } else if (detector == "aruco") {
            output = detectAndDrawArUco(inputImage, maxNumFeatures);
        } else {
            std::println("Unknown detector: {}", detector);
            return EXIT_FAILURE;
        }

        if (doExport)
        {
            // Write image returned from detectAndDraw to outputPath
            cv::imwrite(outputPath.string(), output);
        }
        else
        {
            // Display image returned from detectAndDraw on screen and wait for keypress
            cv::imshow("Detected Features", output);
            cv::waitKey(0);
        }
    }

    if (isVideo)
    {
        cv::VideoWriter outputVideo;
        if (doExport)
        {
            // Open output video for writing using the same fps as the input video
            //       and the codec set to cv::VideoWriter::fourcc('m', 'p', '4', 'v')
            double fps = inputVideo.get(cv::CAP_PROP_FPS);
            cv::Size frameSize(
                (int)inputVideo.get(cv::CAP_PROP_FRAME_WIDTH),
                (int)inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT));

            outputVideo.open(outputPath.string(),
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps,
                            frameSize);

            if (!outputVideo.isOpened()) {
                std::println("Error: Could not open output video for writing.");
                return EXIT_FAILURE;
            }
        }

        while (true)
        {
            // Get next frame from input video
            cv::Mat frame;
            inputVideo >> frame;
            // If frame is empty, break out of the while loop
            if (frame.empty()) {
                break;  // End of video
            }
            cv::Mat output;
            // Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
            if (detector == "harris") {
                output = detectAndDrawHarris(frame, maxNumFeatures);
            } else if (detector == "shi") {
                output = detectAndDrawShiAndTomasi(frame, maxNumFeatures);
            } else if (detector == "fast") {
                output = detectAndDrawFAST(frame, maxNumFeatures);
            } else if (detector == "aruco") {
                output = detectAndDrawArUco(frame, maxNumFeatures);
            } else {
                std::println("Unknown detector: {}", detector);
                return EXIT_FAILURE;
            }

            if (doExport)
            {
                // Write image returned from detectAndDraw to frame of output video
                outputVideo.write(output);
            }
            else
            {
                // Display image returned from detectAndDraw on screen and wait for 1000/fps milliseconds
                cv::imshow("Video Output", output);
                if (cv::waitKey(1000.0 / inputVideo.get(cv::CAP_PROP_FPS)) >= 0) {
                    break;
                }
            }
        }

        // release the input video object
        inputVideo.release();
        if (doExport)
        {
            // release the output video object
            outputVideo.release();
        }
    }

    return EXIT_SUCCESS;
}



