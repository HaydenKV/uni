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
#include <bitset>
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

/*
ChessboardImage:
Detects inner corners r^i_{Q/O} in the image (subpixel refinement), sets isFound.
*/
ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
    : image(image_)
    , filename(filename_)
    , isFound(false)
{
    // Early exit if no image
    if (image.empty()) {
        corners.clear();
        return;
    }

    // Grayscale
    cv::Mat gray;
    if (image.channels() == 3 || image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Corner detection (robust flags)
    std::vector<cv::Point2f> detected;
    const int findFlags =
        cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE |
        cv::CALIB_CB_FAST_CHECK;

    isFound = cv::findChessboardCorners(gray, chessboard.boardSize, detected, findFlags);

    if (isFound) {
        // Subpixel refinement (improves calibration accuracy)
        const cv::Size  winSize(11, 11);
        const cv::Size  zeroZone(-1, -1);
        const auto term = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 1e-3);

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
    const Posed Tnb = camera.cameraToBody(Tnc); //convert 
    const Posed Tcn = Tnc.inverse(); // world -> camera

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

    Posed Tcn(Thetacn, rNCc);
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

/*
Camera::calibrate():
Calibrates intrinsics (K,dist) from multiple chessboard views.
- Object points: gridPoints() in world frame (Z=0 plane).
- Image points: per-image inner corners.
- Flags enable rational, thin-prism, and tilted distortion models.
Post-step: calcFieldOfView() precomputes FoV and an azimuth LUT used in visibility tests.
*/
void Camera::calibrate(ChessboardData & chessboardData)
{
    // 3D planar points
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    // 2D detections per image
    std::vector<std::vector<cv::Point2f>> rQOi_all;
    for (const auto & chessboardImage : chessboardData.chessboardImages)
        rQOi_all.push_back(chessboardImage.corners);
    assert(!rQOi_all.empty());

    // image geometry
    imageSize = chessboardData.chessboardImages[0].image.size();
    
    // distortion model selection
    flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_TILTED_MODEL;
    // flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

    // initialise outputs
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs   = cv::Mat::zeros(12, 1, CV_64F);

    std::vector<cv::Mat> Thetacn_all, rNCc_all;
    double rms;
    std::print("Calibrating camera...");

    // Prepare object points replicated per image
    std::vector<std::vector<cv::Point3f>> objectPoints(rQOi_all.size(), rPNn_all);
    rms = cv::calibrateCamera(objectPoints, rQOi_all, imageSize,
                              cameraMatrix, distCoeffs,
                              Thetacn_all, rNCc_all, flags);
    std::println(" done");
    
    // Pre-compute FOV/LUT used by FOV checks
    calcFieldOfView();

    // Save extrinsics per image: build T^n_C by inverting OpenCV's T^c_n
    assert(chessboardData.chessboardImages.size() == rNCc_all.size());
    assert(chessboardData.chessboardImages.size() == Thetacn_all.size());
    for (std::size_t k = 0; k < chessboardData.chessboardImages.size(); ++k)
    {
        Posed & Tnc = chessboardData.chessboardImages[k].Tnc;
        Posed Tcn(Thetacn_all[k], rNCc_all[k]);
        Tnc = Tcn.inverse();
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

/*
calcFieldOfView():
Computes horizontal/vertical/diagonal FoV by mapping image edges to unit rays
u^c = pixelToVector([u,v]) and taking inter-ray angles.
Builds an azimuth-indexed cos(θ_max+margin) lookup (0..359°) for fast FOV checks.
*/
void Camera::calcFieldOfView()
{
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    const double W = static_cast<double>(imageSize.width);
    const double H = static_cast<double>(imageSize.height);
    const double midx = (W - 1.0) * 0.5;
    const double midy = (H - 1.0) * 0.5;

    auto angle_between = [](const cv::Vec3d& a, const cv::Vec3d& b) {
        double c = a.dot(b) / (cv::norm(a) * cv::norm(b));
        c = std::clamp(c, -1.0, 1.0);
        return std::acos(c);
    };

    // FoV angles from unit vectors on the image border
    cv::Vec3d uLeft  = pixelToVector(cv::Vec2d(0.0,   midy));
    cv::Vec3d uRight = pixelToVector(cv::Vec2d(W-1.0, midy));
    hFOV = angle_between(uLeft, uRight);

    cv::Vec3d uTop    = pixelToVector(cv::Vec2d(midx, 0.0));
    cv::Vec3d uBottom = pixelToVector(cv::Vec2d(midx, H-1.0));
    vFOV = angle_between(uTop, uBottom);

    cv::Vec3d uTL = pixelToVector(cv::Vec2d(0.0,   0.0));
    cv::Vec3d uBR = pixelToVector(cv::Vec2d(W-1.0, H-1.0));
    dFOV = angle_between(uTL, uBR);

    // Azimuth → cos(theta_limit) LUT with small safety margin
    cosThetaLimit_.assign(360, -1.0);
    for (int deg = 0; deg < 360; ++deg) {
        double maxTheta = 0.0;
        double radAz = deg * CV_PI / 180.0;
        for (int rstep = 0; rstep < 2; ++rstep) {
            double r = 0.499 - 0.002 * rstep; // slightly inside
            double u = (W - 1) * (0.5 + r * std::cos(radAz));
            double v = (H - 1) * (0.5 + r * std::sin(radAz));
            cv::Vec3d uPCc = pixelToVector(cv::Vec2d(u, v));
            double theta = std::acos(std::clamp(uPCc[2] / cv::norm(uPCc), -1.0, 1.0));
            maxTheta = std::max(maxTheta, theta);
        }
        cosThetaLimit_[deg] = std::cos(maxTheta + (10.0 * CV_PI / 180.0)); // +10° margin
    }
}

cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Posed & Tnb) const
{
    // Camera pose Tnc (i.e., Rnc, rCNn)
    Posed Tnc = bodyToCamera(Tnb); // Tnb*Tbc

    // World -> Camera
    Posed Tcn = Tnc.inverse();

    // Position of point P in camera coords
    cv::Vec3d rPCc = Tcn * rPNn;

    // Compute the unit vector uPCc from the world position rPNn and camera pose Tnc
    cv::Vec3d uPCc = rPCc / cv::norm(rPCc);

    return uPCc;
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Posed & Tnb) const
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

    cv::Vec3d dir = rPCc / cv::norm(rPCc);
    const cv::Vec3d z(0,0,1);
   
    // azimuth-based check
    double az = std::atan2(dir[1], dir[0]) * 180.0 / CV_PI; // degrees
    int bin = static_cast<int>(std::lround(az));
    bin = (bin % 360 + 360) % 360;                          // wrap to [0,359]
    double cosang = dir.dot(z);
    if (cosang < cosThetaLimit_[bin])
        return false;

    // projection+bounds
    cv::Vec3d P(rPCc[0]/rPCc[2], rPCc[1]/rPCc[2], 1.0);
    cv::Vec2d px = vectorToPixel(P);
    if (!std::isfinite(px[0]) || !std::isfinite(px[1])) return false;

    return (px[0] >= 0.0 && px[0] < imageSize.width &&
            px[1] >= 0.0 && px[1] < imageSize.height);
}


bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Posed & Tnb) const
{
    return isVectorWithinFOV(worldToVector(rPNn, Tnb));
}


/*
Pixel-bound helpers with optional margin (default from CamDefaults::BorderMarginPx).
*/
bool Camera::isPixelInside(const cv::Point2f& uv, int margin) const {
    const int m = (margin < 0 ? CamDefaults::BorderMarginPx : margin);
    const int W = imageSize.width;
    const int H = imageSize.height;
    if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) return false;
    return (uv.x >= m && uv.x <= (W - 1 - m) &&
            uv.y >= m && uv.y <= (H - 1 - m));
}

bool Camera::isPixelInside(const cv::Point2d& uv, int margin) const {
    return isPixelInside(cv::Point2f(static_cast<float>(uv.x),
                                     static_cast<float>(uv.y)), margin);
}

bool Camera::isPixelInside(const Eigen::Vector2d& uv, int margin) const {
    return isPixelInside(cv::Point2f(static_cast<float>(uv.x()),
                                     static_cast<float>(uv.y())), margin);
}

/*
Corner convenience checks for ArUco-style 4-corner bundles.
*/
bool Camera::areCornersInside(const std::array<cv::Point2f,4>& c, int margin) const {
    for (int k = 0; k < 4; ++k) {
        if (!isPixelInside(c[k], margin)) return false;
    }
    return true;
}

bool Camera::areCornersInside(const Eigen::Matrix<double,8,1>& uv8, int margin) const {
    for (int k = 0; k < 4; ++k) {
        const double u = uv8(2*k);
        const double v = uv8(2*k + 1);
        if (!isPixelInside(cv::Point2f(static_cast<float>(u), static_cast<float>(v)), margin))
            return false;
    }
    return true;
}

/*
isVectorWithinFOVConservative():
Adds more pixel-margin gating to isVectorWithinFOV().
*/
bool Camera::isVectorWithinFOVConservative(const cv::Vec3d& rPCc, int margin) const {
    if (!isVectorWithinFOV(rPCc)) return false;

    if (rPCc[2] <= 1e-9) return false;
    cv::Vec3d P(rPCc[0]/rPCc[2], rPCc[1]/rPCc[2], 1.0);
    cv::Vec2d px = vectorToPixel(P);
    return isPixelInside(cv::Point2f(static_cast<float>(px[0]),
                                     static_cast<float>(px[1])), margin);
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

