#include <cassert>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <format>
#include <vector>
#include <filesystem>
#include <regex>
#include <print>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include "to_string.hpp"
#include "rotation.hpp"
#include "Pose.hpp"
#include "Camera.h"

void Chessboard::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "grid_width"  << boardSize.width
       << "grid_height" << boardSize.height
       << "square_size" << squareSize
       << "}";
}

void Chessboard::read(const cv::FileNode & node)
{
    node["grid_width"]  >> boardSize.width;
    node["grid_height"] >> boardSize.height;
    node["square_size"] >> squareSize;
}

std::vector<cv::Point3f> Chessboard::gridPoints() const
{
    std::vector<cv::Point3f> rPNn_all;
    rPNn_all.reserve(boardSize.height*boardSize.width);
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            rPNn_all.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));   
    return rPNn_all; 
}

std::ostream & operator<<(std::ostream & os, const Chessboard & chessboard)
{
    return os << "boardSize: " << chessboard.boardSize << ", squareSize: " << chessboard.squareSize;
}

ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
    : image(image_)
    , filename(filename_)
    , isFound(false)
{
    // TODO: Merge from Lab 4
}

void ChessboardImage::drawCorners(const Chessboard & chessboard)
{
    cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
}

void ChessboardImage::drawBox(const Chessboard & chessboard, const Camera & camera)
{
    // TODO: Merge from Lab 4
}

void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
{
    std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

    cv::Mat Thetacn, rNCc;
    cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);

    Pose<double> Tcn(Thetacn, rNCc);
    Tnc = Tcn.inverse();
}

ChessboardData::ChessboardData(const std::filesystem::path & configPath)
{
    // Ensure the config file exists
    if (!std::filesystem::exists(configPath))
    {
        throw std::runtime_error("Config file does not exist: " + configPath.string());
    }

    // Open the config file
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        throw std::runtime_error("Failed to open config file: " + configPath.string());
    }

    // Read chessboard configuration
    cv::FileNode node = fs["chessboard_data"];
    node["chessboard"] >> chessboard;
    std::println("Chessboard: {}", to_string(chessboard));

    // Read file pattern for chessboard images
    std::string pattern;
    node["file_regex"] >> pattern;
    fs.release();

    // Create regex object from pattern
    std::regex re(pattern, std::regex_constants::basic | std::regex_constants::icase);
    
    // Get the directory containing the config file
    std::filesystem::path root = configPath.parent_path();
    std::println("Scanning directory {} for file pattern \"{}\"", root.string(), pattern);

    // Populate chessboard images from regex
    chessboardImages.clear();
    if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
    {
        // Iterate through all files in the directory and its subdirectories
        for (const auto & p : std::filesystem::recursive_directory_iterator(root))
        {
            if (std::filesystem::is_regular_file(p))
            {
                // Check if the file matches the regex pattern
                if (std::regex_match(p.path().filename().string(), re))
                {
                    std::print("Loading {}...", p.path().filename().string());

                    // Try to load the file as an image
                    cv::Mat image = cv::imread(p.path().string(), cv::IMREAD_COLOR);

                    bool isImage = !image.empty();
                    if (isImage)
                    {
                        // If it's an image, detect chessboard
                        std::print(" done, detecting chessboard...");
                        ChessboardImage ci(image, chessboard, p.path().filename());
                        std::println("{}", ci.isFound ? " found" : " not found");
                        if (ci.isFound)
                        {
                            chessboardImages.push_back(ci);
                        }
                    }
                    else
                    {
                        // If it's not an image, try to load it as a video
                        cv::VideoCapture cap(p.path().string());
                        bool isVideo = cap.isOpened();
                        if (isVideo)
                        {
                            // Get number of video frames
                            int nFrames = 0; // TODO: Merge from Lab 4
                            std::println(" done, found {} frames", nFrames);

                            // Loop through selected frames
                            for (int idxFrame = 0; idxFrame < nFrames; /*TODO: Merge from Lab 4*/)
                            {
                                // Read frame
                                std::print("Reading {} frame {}...", p.path().filename().string(), idxFrame);
                                cv::Mat frame;
                                // TODO: Merge from Lab 4

                                if (frame.empty())
                                {
                                    std::println(" end of file found");
                                    break;
                                }

                                // Detect chessboard in frame
                                std::print(" done, detecting chessboard...");
                                std::string baseName = p.path().stem().string();
                                std::string frameFilename = std::format("{}_{:05d}.jpg", baseName, idxFrame);
                                ChessboardImage ci(frame, chessboard, frameFilename);
                                std::println("{}", ci.isFound ? " found" : " not found");
                                if (ci.isFound)
                                {
                                    chessboardImages.push_back(ci);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ChessboardData::drawCorners()
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawCorners(chessboard);
    }
}

void ChessboardData::drawBoxes(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawBox(chessboard, camera);
    }
}

void ChessboardData::recoverPoses(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.recoverPose(chessboard, camera);
    }
}

void Camera::calibrate(ChessboardData & chessboardData)
{
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    std::vector<std::vector<cv::Point2f>> rQOi_all;
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        rQOi_all.push_back(chessboardImage.corners);
    }
    assert(!rQOi_all.empty());

    imageSize = chessboardData.chessboardImages[0].image.size();
    
    flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

    // Find intrinsic and extrinsic camera parameters
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
    std::vector<cv::Mat> Thetacn_all, rNCc_all;
    double rms;
    std::print("Calibrating camera...");
    // TODO: Merge from Lab 4
    std::println(" done");
    
    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    // Write extrinsic camera parameters for each chessboard image
    assert(chessboardData.chessboardImages.size() == rNCc_all.size());
    assert(chessboardData.chessboardImages.size() == Thetacn_all.size());
    for (std::size_t k = 0; k < chessboardData.chessboardImages.size(); ++k)
    {
        // Set the camera orientation and position (extrinsic camera parameters)
        Pose<double> & Tnc = chessboardData.chessboardImages[k].Tnc;
        // TODO: Merge from Lab 4
        // Tnc.rotationMatrix = ???;            // Rnc
        // Tnc.translationVector = ???;         // rCNn
    }
    
    printCalibration();
    std::println("{:>30} {}", "RMS reprojection error:", rms);

    assert(cv::checkRange(cameraMatrix));
    assert(cv::checkRange(distCoeffs));
}

void Camera::printCalibration() const
{
    std::bitset<8*sizeof(flags)> bitflag(flags);
    std::println("\nCalibration data:");
    std::println("{:>30} {}", "Bit flags:", bitflag.to_string());
    std::println("{:>30}\n{}", "cameraMatrix:", to_string(cameraMatrix));
    std::println("{:>30}\n{}", "distCoeffs:", to_string(distCoeffs.t()));
    std::println("{:>30} (fx, fy) = ({}, {})", "Focal lengths:",
              cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1));       
    std::println("{:>30} (cx, cy) = ({}, {})", "Principal point:",
              cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2));     
    std::println("{:>30} {} deg", "Field of view (horizontal):", 180.0/CV_PI*hFOV);
    std::println("{:>30} {} deg", "Field of view (vertical):", 180.0/CV_PI*vFOV);
    std::println("{:>30} {} deg", "Field of view (diagonal):", 180.0/CV_PI*dFOV);
}

void Camera::calcFieldOfView()
{
    // TODO: Merge from Lab 4
}

cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    cv::Vec3d uPCc;
    // TODO: Merge from Lab 4
    return uPCc;
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    return vectorToPixel(worldToVector(rPNn, Tnb));
}

cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const
{
    cv::Vec2d rQOi;
    // TODO: Merge from Lab 4
    return rQOi;
}

Eigen::Vector2d Camera::vectorToPixel(const Eigen::Vector3d & rPCc, Eigen::Matrix23d & J) const
{
    Eigen::Vector2d rQOi;
    // TODO: Lab 8 (optional)
    return rQOi;
}

cv::Vec3d Camera::pixelToVector(const cv::Vec2d & rQOi) const
{
    cv::Vec3d uPCc;
    // TODO: Merge from Lab 4
    return uPCc;
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    // TODO: Merge from Lab 4
    return false;
}

bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    return isVectorWithinFOV(worldToVector(rPNn, Tnb));
}

Eigen::Matrix<double, 2, Eigen::Dynamic> Camera::undistort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOi) const
{
    // Convert from Eigen matrix to std::vector of cv::Point2d
    std::vector<cv::Point2d> rQOi_cv(rQOi.cols());
    for (int i = 0; i < rQOi.cols(); ++i)
    {
        rQOi_cv[i] = cv::Point2d(rQOi(0, i), rQOi(1, i));
    }

    // Undistort points
    std::vector<cv::Point2d> rQbarOi_cv;
    // TODO: Lab 10

    // Convert from std::vector of cv::Point2d to Eigen matrix
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOi(2, rQOi.cols());
    for (int i = 0; i < rQbarOi_cv.size(); ++i)
    {
        rQbarOi(0, i) = rQbarOi_cv[i].x;
        rQbarOi(1, i) = rQbarOi_cv[i].y;
    }

    return rQbarOi;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> Camera::undistort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQOi) const
{
    // Extract the Euclidean points from the homogeneous coordinates
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi = pQOi.topRows<2>().array().rowwise() / pQOi.row(2).array();

    // Call the Euclidean undistort function
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOi = undistort(rQOi);

    // Create the output matrix with homogeneous coordinates
    Eigen::Matrix<double, 3, Eigen::Dynamic> pQbarOi(3, pQOi.cols());
    pQbarOi.topRows<2>() = rQbarOi;
    pQbarOi.row(2).setOnes();

    return pQbarOi;
}

Eigen::Matrix<double, 2, Eigen::Dynamic> Camera::distort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQbarOi) const
{
    double fx = cameraMatrix.at<double>( 0,  0);
    double fy = cameraMatrix.at<double>( 1,  1);
    double cx = cameraMatrix.at<double>( 0,  2);
    double cy = cameraMatrix.at<double>( 1,  2);

    // Convert from Euclidean coordinates to homogeneous coordinates
    Eigen::Matrix<double, 3, Eigen::Dynamic> pQbarOi(3, rQbarOi.cols());
    pQbarOi.topRows<2>() = rQbarOi;
    pQbarOi.row(2).setOnes();

    // Solve K*rPCc = pQbarOi for rPCc
    Eigen::Matrix<double, 3, Eigen::Dynamic> rPCc;
    // TODO: Lab 10

    // Use camera model (with lens distortion) to get pixel coordinates
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi;
    // TODO: Lab 10
    return rQOi;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> Camera::distort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQbarOi) const
{
    // Convert from homogeneous coordinates to Euclidean coordinates
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOi = pQbarOi.topRows<2>().array().rowwise() / pQbarOi.row(2).array();

    // Call the Euclidean distort function
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi = distort(rQbarOi);

    // Convert from Euclidean coordinates to homogeneous coordinates
    Eigen::Matrix<double, 3, Eigen::Dynamic> pQOi(3, pQbarOi.cols());
    pQOi.topRows<2>() = rQOi;
    pQOi.row(2).setOnes();

    return pQOi;
}

void Camera::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "camera_matrix"           << cameraMatrix
       << "distortion_coefficients" << distCoeffs
       << "flags"                   << flags
       << "imageSize"               << imageSize
       << "}";
}

void Camera::read(const cv::FileNode & node)
{
    node["camera_matrix"]           >> cameraMatrix;
    node["distortion_coefficients"] >> distCoeffs;
    node["flags"]                   >> flags;
    node["imageSize"]               >> imageSize;

    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.type() == CV_64F);
    assert(distCoeffs.cols == 1);
    assert(distCoeffs.type() == CV_64F);
}

