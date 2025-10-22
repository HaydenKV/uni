#include <cassert>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <format>
#include <vector>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <print>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include "to_string.hpp"
#include "Camera.h"

// Default constructor (R = I, r = 0)
Pose::Pose()
    : rotationMatrix(cv::Matx33d::eye())
    , translationVector(cv::Vec3d::zeros())
{}

// Constructor from OpenCV rotation vector and translation vector
Pose::Pose(const cv::Mat & rvec, const cv::Mat & tvec)
{
    cv::Rodrigues(rvec, rotationMatrix);
    translationVector = tvec;
}

// Group operation of SE(3)
//
// Tac = Tab*Tbc
//
// [ Rac, rCAa ] = [ Rab, rBAa ] * [ Rbc, rCBb ] = [ Rab*Rbc, Rab*rCBb + rBAa ]
// [   0,    1 ]   [   0,    1 ]   [   0,    1 ]   [       0,               1 ]
//
Pose Pose::operator*(const Pose & other) const
{
    Pose result;
    result.rotationMatrix = rotationMatrix * other.rotationMatrix;
    result.translationVector = rotationMatrix * other.translationVector + translationVector;
    return result;
}

// Action of SE(3) on P3
//
// Point alias: pPAa = Tab*pPBb
// [ rPAa ] = [ Rab, rBAa ] * [ rPBb ] = [ Rab*rPBb + rBAa ] 
// [    1 ]   [   0,    1 ]   [    1 ]   [               1 ]
//
// Point alibi: pBPa = Tab*pAPb
// [ rBPa ] = [ Rab, rBAa ] * [ rAPb ] = [ Rab*rAPb + rBAa ] 
// [    1 ]   [   0,    1 ]   [    1 ]   [               1 ]
//
cv::Vec3d Pose::operator*(const cv::Vec3d & r) const
{
    return rotationMatrix * r + translationVector;
}

// Inverse element in SE(3)
//
// Tab^-1 = Tba
//
// [ Rab, rBAa ]^-1 = [ Rba, rABb ] = [ Rab^T, -Rab^T*rBAa ]
// [   0,    1 ]      [   0,    1 ]   [     0,           1 ]
//
Pose Pose::inverse() const
{
    Pose result;
    result.rotationMatrix = rotationMatrix.t();
    result.translationVector = -result.rotationMatrix * translationVector;
    return result;
}

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
    // TODO:
    //  - Detect chessboard corners in image and set the corners member
    //  - (optional) Do subpixel refinement of detected corners

    // --- Detect chessboard corners
    if (image.empty()) {
        corners.clear();
        return;
    }

    // 1) grayscale
    cv::Mat gray;
    if (image.channels() == 3 || image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // 2) Find inner-corner grid (row-major order)
    //    Use flags that improve robustness across lighting/contrast.
    std::vector<cv::Point2f> detected;
    const int findFlags =
        cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE |
        cv::CALIB_CB_FAST_CHECK;   // quick reject for frames with no board

    isFound = cv::findChessboardCorners(gray, chessboard.boardSize, detected, findFlags);

    if (isFound) {
        // 3) Subpixel refinement for accuracy
        //    Tight termination criteria is important for good calibration.
        const cv::Size  winSize(11, 11);
        const cv::Size  zeroZone(-1, -1);
        const auto term = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 1e-3);

        // OpenCV expects an 8-bit grayscale image for cornerSubPix
        cv::Mat gray8;
        if (gray.type() != CV_8U) {
            gray.convertTo(gray8, CV_8U);
        } else {
            gray8 = gray;
        }
        cv::cornerSubPix(gray8, detected, winSize, zeroZone, term);

        corners = std::move(detected);
    } else {
        corners.clear();
    }

}

void ChessboardImage::drawCorners(const Chessboard & chessboard)
{
    cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
}

void ChessboardImage::drawBox(const Chessboard & chessboard, const Camera & camera)
{
    if (!isFound) return;

    // Tnc (camera pose in world)
    const Pose Tnb = camera.cameraToBody(Tnc); //convert 
    const Pose Tcn = Tnc.inverse(); // world -> camera

    // dimensions (-1 for inner)
    const double W = (chessboard.boardSize.width  - 1) * chessboard.squareSize;
    const double H = (chessboard.boardSize.height - 1) * chessboard.squareSize;

    // 8 base corners in world (N)
    const cv::Vec3d p0(0.0, 0.0, 0.0);
    const cv::Vec3d p1(W  , 0.0, 0.0);
    const cv::Vec3d p2(W  , H  , 0.0);
    const cv::Vec3d p3(0.0, H  , 0.0);

    // Decide extrusion direction automatically so the box rises toward the camera
    const double Zmag = 0.23;                // metres
    const cv::Vec3d c = 0.5 * (p0 + p2);     // board centre
    const double z_up   = (Tcn * (c + cv::Vec3d(0,0, Zmag)))[2];
    const double z_down = (Tcn * (c + cv::Vec3d(0,0,-Zmag)))[2];
    const double Z = (z_up < z_down) ? Zmag : -Zmag;

    // Lid corners
    const cv::Vec3d q0 = p0 + cv::Vec3d(0,0,Z);
    const cv::Vec3d q1 = p1 + cv::Vec3d(0,0,Z);
    const cv::Vec3d q2 = p2 + cv::Vec3d(0,0,Z);
    const cv::Vec3d q3 = p3 + cv::Vec3d(0,0,Z);

    // Colours (BGR)
    const cv::Scalar BLUE (255,   0,   0);
    const cv::Scalar RED  (  0,   0, 255);
    const cv::Scalar GREEN(  0, 255,   0);

    // Draw a 3D edge a->b as short segments so curves follow lens distortion
    auto draw_edge = [&](const cv::Vec3d& a, const cv::Vec3d& b, const cv::Scalar& color, int thickness)
    {
        constexpr int kSegs = 64; //48; // segment divided by this
        const double maxJump = 60.0; // prevents draw if 2 consecutive points are greater than X appart
        cv::Point prev;
        bool havePrev = false;

        for (int i = 0; i <= kSegs; ++i)
        {
            const double t = static_cast<double>(i) / kSegs;
            const cv::Vec3d P = a*(1.0 - t) + b*t;

            // check if within FOV, "break" if doesnt connect
            if (!camera.isWorldWithinFOV(P, Tnb)) { havePrev = false; continue; }

            // convert 3d to 2d using camera intrinscis
            const cv::Vec2d pix = camera.worldToPixel(P, Tnb);
            const cv::Point cur(cvRound(pix[0]), cvRound(pix[1]));

            if (!havePrev) {                            // first visible sample
                prev = cur;
                havePrev = true;
                continue;
            }

            // avoid wraparound / re-entry across the border
            if (cv::norm(cur - prev) > maxJump) {
                prev = cur;                             // restart a new segment
                continue;
            }

            // draw the segment
            cv::line(image, prev, cur, color, thickness, cv::LINE_AA);
            prev = cur;   
        }
    };

    // Base (blue)
    draw_edge(p0, p1, BLUE, 2);  draw_edge(p1, p2, BLUE, 2);
    draw_edge(p2, p3, BLUE, 2);  draw_edge(p3, p0, BLUE, 2);

    // Uprights (red)
    draw_edge(p0, q0, RED, 2);   draw_edge(p1, q1, RED, 2);
    draw_edge(p2, q2, RED, 2);   draw_edge(p3, q3, RED, 2);

    // Lid (green, a touch thicker)
    draw_edge(q0, q1, GREEN, 2); draw_edge(q1, q2, GREEN, 2);
    draw_edge(q2, q3, GREEN, 2); draw_edge(q3, q0, GREEN, 2);
}

void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
{
    std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

    cv::Mat Thetacn, rNCc;
    cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);

    Pose Tcn(Thetacn, rNCc);
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
                            int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
                            if (nFrames <= 0)
                            {
                                // Fallback: manually count then rewind (some backends can’t report frame count)
                                nFrames = 0;
                                cv::Mat tmp;
                                while (cap.read(tmp)) ++nFrames;
                                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                            }
                            std::println(" done, found {} frames", nFrames);

                            if (nFrames <= 0)
                            {
                                std::println(" no frames readable, skipping");
                                continue;
                            }

                            // Select a subset of frames for calibration
                            const int target  = 50; 
                            const int stride   = std::max(1, nFrames / (target)); //*2 returns 1 not found! larger bottom is more distance between frames used (more varitey)
                            int kept = 0;
                            
                            // Loop through selected frames
                            for (int idxFrame = 0; idxFrame < nFrames; idxFrame += stride)
                            {
                                // Read frame
                                std::print("Reading {} frame {}...", p.path().filename().string(), idxFrame);
                                cv::Mat frame;

                                // Seek to target frame then read
                                cap.set(cv::CAP_PROP_POS_FRAMES, idxFrame);
                                if (!cap.read(frame))
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
                                    if (++kept >= target)
                                    {
                                        std::println(" reached target of {} detections, stopping sampling", target);
                                        break;
                                    }
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
    // get 3d planar points 
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    std::vector<std::vector<cv::Point2f>> rQOi_all;
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        rQOi_all.push_back(chessboardImage.corners); // collect 2D points per image
    }
    assert(!rQOi_all.empty());

    // needed by solver
    imageSize = chessboardData.chessboardImages[0].image.size();
    
    // distortion model
    // increased precision in corners with tilted model included
    flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_TILTED_MODEL;

    // Find intrinsic and extrinsic camera parameters
    // initialise and allocate space
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
    // outputs per image rotations and translations
    std::vector<cv::Mat> Thetacn_all, rNCc_all;
    double rms;
    std::print("Calibrating camera...");
    // TODO: Calibrate camera from detected chessboard corners
    std::vector<std::vector<cv::Point3f>> objectPoints(rQOi_all.size(), rPNn_all);
        rms = cv::calibrateCamera(
        objectPoints,            // 3D points per image
        rQOi_all,                 // 2D detected corners per image
        imageSize,
        cameraMatrix,
        distCoeffs,
        Thetacn_all,              // output rvecs (Rcn)
        rNCc_all,                 // output tvecs (rNCc)
        flags
    );
    std::println(" done");
    
    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    // Write extrinsic camera parameters for each chessboard image
    assert(chessboardData.chessboardImages.size() == rNCc_all.size());
    assert(chessboardData.chessboardImages.size() == Thetacn_all.size());
    for (std::size_t k = 0; k < chessboardData.chessboardImages.size(); ++k)
    {
        // Set the camera orientation and position (extrinsic camera parameters)
        Pose & Tnc = chessboardData.chessboardImages[k].Tnc;
        // TODO
        // cv::Mat Rcn;
        // cv::Rodrigues(Thetacn_all[k], Rcn); // Convert rvec to 3x3 rotation matrix
        // // Convert to our convention: camera pose in world
        // cv::Mat Rnc = Rcn.t(); // inverse rotation
        // cv::Mat rCNn = -Rnc * rNCc_all[k]; // transform translation
        // Tnc.rotationMatrix = Rnc;            // Rnc
        // Tnc.translationVector = cv::Vec3d(rCNn);         // rCNn

        // Build world->camera from OpenCV’s rvec/tvec, then invert to camera->world
        // From function below that does the math
        Pose Tcn(Thetacn_all[k], rNCc_all[k]); // function top of script
        Tnc = Tcn.inverse(); // cam to world
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
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    // TODO:
    const double W = static_cast<double>(imageSize.width);
    const double H = static_cast<double>(imageSize.height);
    // centre
    const double midx = (W - 1.0) * 0.5;
    const double midy = (H - 1.0) * 0.5;

    // helper function (angle between 2 vecs)
    auto angle_between = [](const cv::Vec3d& a, const cv::Vec3d& b) {
        double c = a.dot(b) / (cv::norm(a) * cv::norm(b));
        c = std::clamp(c, -1.0, 1.0);
        return std::acos(c);
    };

    // Horizontal FOV: left vs right at image mid-row
    cv::Vec3d uLeft  = pixelToVector(cv::Vec2d(0.0,   midy));
    cv::Vec3d uRight = pixelToVector(cv::Vec2d(W-1.0, midy));
    hFOV = angle_between(uLeft, uRight);

    // Vertical FOV: top vs bottom at image mid-col
    cv::Vec3d uTop    = pixelToVector(cv::Vec2d(midx, 0.0));
    cv::Vec3d uBottom = pixelToVector(cv::Vec2d(midx, H-1.0));
    vFOV = angle_between(uTop, uBottom);

    // Diagonal FOV: top-left vs bottom-right
    cv::Vec3d uTL = pixelToVector(cv::Vec2d(0.0,   0.0));
    cv::Vec3d uBR = pixelToVector(cv::Vec2d(W-1.0, H-1.0));
    dFOV = angle_between(uTL, uBR);

    // --- azimuth-based theta limit table ---
    cosThetaLimit_.assign(360, -1.0);
    for (int deg = 0; deg < 360; ++deg) {
        double maxTheta = 0.0;
        double radAz = deg * CV_PI / 180.0;
        // sample along border in this azimuth direction
        for (int rstep = 0; rstep < 2; ++rstep) {
            double r = 0.499 - 0.002 * rstep; // slightly inside
            double u = (W - 1) * (0.5 + r * std::cos(radAz));
            double v = (H - 1) * (0.5 + r * std::sin(radAz));
            cv::Vec3d uPCc = pixelToVector(cv::Vec2d(u, v));
            double theta = std::acos(std::clamp(uPCc[2] / cv::norm(uPCc), -1.0, 1.0));
            maxTheta = std::max(maxTheta, theta);
        }
        cosThetaLimit_[deg] = std::cos(maxTheta + (10.0 * CV_PI / 180.0)); // margin
    }
}

Pose Camera::cameraToBody(const Pose & Tnc) const
{
    // Tnb = Tnc*Tcb
    return Tnc*Tbc.inverse();
}

Pose Camera::bodyToCamera(const Pose & Tnb) const
{
    // Tnc = Tnb*Tbc
    return Tnb*Tbc;
}

cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose & Tnb) const
{
    // Camera pose Tnc (i.e., Rnc, rCNn)
    Pose Tnc = bodyToCamera(Tnb); // Tnb*Tbc

    // World -> Camera
    Pose Tcn = Tnc.inverse();

    // Position of point P in camera coords
    cv::Vec3d rPCc = Tcn * rPNn;

    // Compute the unit vector uPCc from the world position rPNn and camera pose Tnc
    cv::Vec3d uPCc = rPCc / cv::norm(rPCc);
    // TODO
    return uPCc;
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose & Tnb) const
{
    return vectorToPixel(worldToVector(rPNn, Tnb));
}

cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const
{
    cv::Point3f P(static_cast<float>(rPCc[0]/rPCc[2]),
                  static_cast<float>(rPCc[1]/rPCc[2]),
                  1.0f);

    std::vector<cv::Point3f> obj{P};
    std::vector<cv::Point2f> img;

    cv::Mat rvec = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1,CV_64F);

    cv::projectPoints(obj, rvec, tvec, cameraMatrix, distCoeffs, img);

    return cv::Vec2d(img[0].x, img[0].y);
}

cv::Vec3d Camera::pixelToVector(const cv::Vec2d & rQOi) const
{
    // Convert the input pixel location to undistorted normalized image coordinates
    std::vector<cv::Point2f> distortedPoints;
    distortedPoints.emplace_back(
        static_cast<float>(rQOi[0]),
        static_cast<float>(rQOi[1])
    );

    std::vector<cv::Point2f> undistortedPoints;
    cv::undistortPoints(distortedPoints, undistortedPoints, cameraMatrix, distCoeffs);

    // Form the 3D vector in camera coordinates (z = 1 for normalized plane)
    cv::Vec3d rPCc(
        undistortedPoints[0].x,
        undistortedPoints[0].y,
        1.0
    );

    // Return the unit vector uPCc
    cv::Vec3d uPCc = rPCc / cv::norm(rPCc);
    return uPCc;
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    // finite & in front
    if (!std::isfinite(rPCc[0]) || !std::isfinite(rPCc[1]) || !std::isfinite(rPCc[2])) return false;
    if (rPCc[2] <= 1e-9) return false;

    // Potential solutions ---------------------vv

    cv::Vec3d dir = rPCc / cv::norm(rPCc);
    const cv::Vec3d z(0,0,1);
   
    // --- azimuth-based check
    // Compares angle of direction vector to optical axis using a lookup table
    // cosThetaLimit_[bin] gives the minimum allowed cos(angle) at this azimuth
    double az = std::atan2(dir[1], dir[0]) * 180.0 / CV_PI; // deg
    int bin = static_cast<int>(std::lround(az));
    bin = (bin % 360 + 360) % 360; // wrap to [0,359]
    double cosang = dir.dot(z);
    if (cosang < cosThetaLimit_[bin])
        return false;
    // --- end azimuth-based check ---

    
    // works but bit bendy in edges
    // define forward axis for angle checks
    // const cv::Vec3d z(0,0,1); 
    // double cosang = (P.dot(z)) / (cv::norm(P)*1.0); // angle between ray and optical axis
    // cosang = std::clamp(cosang, -1.0, 1.0);
    // double ang = std::acos(cosang);
    // if (ang > 0.5*dFOV + 1.0*CV_PI/180.0) return false; // 2° margin

    // This is too aggressive because large FOV
    // Frustum test (use half FOV) (for pinhole *flatter image)
    // const double tan_h = std::tan(0.5 * hFOV);
    // const double tan_v = std::tan(0.5 * vFOV);
    // if (std::abs(P[0]) > tan_h) return false; // |x/z| > tan(hFOV/2)
    // if (std::abs(P[1]) > tan_v) return false; // |y/z| > tan(vFOV/2)

    // Potential solutions ---------------------^^

    // normalize to z=1 (projection scale-invariant so normalizing makes math stable? idk if useful)
    cv::Vec3d P(rPCc[0]/rPCc[2], rPCc[1]/rPCc[2], 1.0);
    // project ray with distortion to pixel space and reject if not finite (maps to valid pixel)
    cv::Vec2d px = vectorToPixel(P);
    if (!std::isfinite(px[0]) || !std::isfinite(px[1])) return false;

    // In-bounds (can add margin here)
    return (px[0] >= 0.0 && px[0] < imageSize.width &&
            px[1] >= 0.0 && px[1] < imageSize.height);
}


bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose & Tnb) const
{
    return isVectorWithinFOV(worldToVector(rPNn, Tnb));
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

