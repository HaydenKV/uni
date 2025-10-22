#include <cassert>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <format>
#include <bitset>
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

// ======================================================
// Pose
// ======================================================

Pose::Pose()
    : rotationMatrix(cv::Matx33d::eye())
    , translationVector(cv::Vec3d::zeros())
{}

Pose::Pose(const cv::Mat & rvec, const cv::Mat & tvec)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    rotationMatrix = R;
    translationVector = tvec;
}

Pose Pose::operator*(const Pose & other) const
{
    Pose result;
    result.rotationMatrix = rotationMatrix * other.rotationMatrix;
    result.translationVector = rotationMatrix * other.translationVector + translationVector;
    return result;
}

cv::Vec3d Pose::operator*(const cv::Vec3d & r) const
{
    return rotationMatrix * r + translationVector;
}

Pose Pose::inverse() const
{
    Pose result;
    result.rotationMatrix = rotationMatrix.t();
    result.translationVector = -result.rotationMatrix * translationVector;
    return result;
}

// ======================================================
// Chessboard
// ======================================================

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

// ======================================================
// ChessboardImage
// ======================================================

ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
    : image(image_)
    , filename(filename_)
    , isFound(false)
{
    if (image.empty()) {
        corners.clear();
        return;
    }

    cv::Mat gray;
    if (image.channels() == 3 || image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    std::vector<cv::Point2f> detected;
    const int findFlags =
        cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE |
        cv::CALIB_CB_FAST_CHECK;

    isFound = cv::findChessboardCorners(gray, chessboard.boardSize, detected, findFlags);

    if (isFound) {
        const cv::Size  winSize(11, 11);
        const cv::Size  zeroZone(-1, -1);
        const auto term = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 1e-3);

        cv::Mat gray8;
        if (gray.type() != CV_8U) gray.convertTo(gray8, CV_8U);
        else                       gray8 = gray;

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

    const Pose Tnb = camera.cameraToBody(Tnc); // convert
    const Pose Tcn = Tnc.inverse();            // world -> camera

    const double W = (chessboard.boardSize.width  - 1) * chessboard.squareSize;
    const double H = (chessboard.boardSize.height - 1) * chessboard.squareSize;

    const cv::Vec3d p0(0.0, 0.0, 0.0);
    const cv::Vec3d p1(W  , 0.0, 0.0);
    const cv::Vec3d p2(W  , H  , 0.0);
    const cv::Vec3d p3(0.0, H  , 0.0);

    const double Zmag = 0.23;                // metres
    const cv::Vec3d c = 0.5 * (p0 + p2);     // board centre
    const double z_up   = (Tcn * (c + cv::Vec3d(0,0, Zmag)))[2];
    const double z_down = (Tcn * (c + cv::Vec3d(0,0,-Zmag)))[2];
    const double Z = (z_up < z_down) ? Zmag : -Zmag;

    const cv::Vec3d q0 = p0 + cv::Vec3d(0,0,Z);
    const cv::Vec3d q1 = p1 + cv::Vec3d(0,0,Z);
    const cv::Vec3d q2 = p2 + cv::Vec3d(0,0,Z);
    const cv::Vec3d q3 = p3 + cv::Vec3d(0,0,Z);

    const cv::Scalar BLUE (255,   0,   0);
    const cv::Scalar RED  (  0,   0, 255);
    const cv::Scalar GREEN(  0, 255,   0);

    auto draw_edge = [&](const cv::Vec3d& a, const cv::Vec3d& b, const cv::Scalar& color, int thickness)
    {
        constexpr int kSegs = 64;
        const double maxJump = 60.0;
        cv::Point prev;
        bool havePrev = false;

        for (int i = 0; i <= kSegs; ++i)
        {
            const double t = static_cast<double>(i) / kSegs;
            const cv::Vec3d P = a*(1.0 - t) + b*t;

            if (!camera.isWorldWithinFOV(P, Tnb)) { havePrev = false; continue; }

            const cv::Vec2d pix = camera.worldToPixel(P, Tnb);
            const cv::Point cur(cvRound(pix[0]), cvRound(pix[1]));

            if (!havePrev) {
                prev = cur;
                havePrev = true;
                continue;
            }

            if (cv::norm(cur - prev) > maxJump) {
                prev = cur;
                continue;
            }

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

    // Lid (green)
    draw_edge(q0, q1, GREEN, 2); draw_edge(q1, q2, GREEN, 2);
    draw_edge(q2, q3, GREEN, 2); draw_edge(q3, q0, GREEN, 2);
}

void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
{
    std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

    cv::Mat Thetacn, rNCc;
    if (camera.useFisheye()) {
        // Fisheye: undistort to pixel space with P = K, then PnP with zero dist
        std::vector<cv::Point2f> undist_pix;
        cv::fisheye::undistortPoints(
            corners, undist_pix,
            camera.cameraMatrix, camera.distCoeffs,
            cv::noArray(), camera.cameraMatrix
        );
        cv::Mat zeroDist;
        cv::solvePnP(rPNn_all, undist_pix, camera.cameraMatrix, zeroDist, Thetacn, rNCc);
    } else {
        // Pinhole
        cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);
    }

    Pose Tcn(Thetacn, rNCc);
    Tnc = Tcn.inverse();
}

// ======================================================
// ChessboardData
// ======================================================

ChessboardData::ChessboardData(const std::filesystem::path & configPath)
{
    if (!std::filesystem::exists(configPath))
        throw std::runtime_error("Config file does not exist: " + configPath.string());

    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Failed to open config file: " + configPath.string());

    cv::FileNode node = fs["chessboard_data"];
    node["chessboard"] >> chessboard;
    std::println("Chessboard: {}", to_string(chessboard));

    std::string pattern;
    node["file_regex"] >> pattern;
    fs.release();

    std::regex re(pattern, std::regex_constants::basic | std::regex_constants::icase);

    std::filesystem::path root = configPath.parent_path();
    std::println("Scanning directory {} for file pattern \"{}\"", root.string(), pattern);

    chessboardImages.clear();
    if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
    {
        for (const auto & p : std::filesystem::recursive_directory_iterator(root))
        {
            if (!std::filesystem::is_regular_file(p)) continue;

            if (std::regex_match(p.path().filename().string(), re))
            {
                std::print("Loading {}...", p.path().filename().string());

                cv::Mat image = cv::imread(p.path().string(), cv::IMREAD_COLOR);

                if (!image.empty())
                {
                    std::print(" done, detecting chessboard...");
                    ChessboardImage ci(image, chessboard, p.path().filename());
                    std::println("{}", ci.isFound ? " found" : " not found");
                    if (ci.isFound) chessboardImages.push_back(ci);
                }
                else
                {
                    cv::VideoCapture cap(p.path().string());
                    if (cap.isOpened())
                    {
                        int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
                        if (nFrames <= 0)
                        {
                            nFrames = 0; cv::Mat tmp;
                            while (cap.read(tmp)) ++nFrames;
                            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                        }
                        std::println(" done, found {} frames", nFrames);
                        if (nFrames <= 0) { std::println(" no frames readable, skipping"); continue; }

                        const int target  = 50;
                        const int stride  = std::max(1, nFrames / target);
                        int kept = 0;

                        for (int idxFrame = 0; idxFrame < nFrames; idxFrame += stride)
                        {
                            std::print("Reading {} frame {}...", p.path().filename().string(), idxFrame);
                            cv::Mat frame;

                            cap.set(cv::CAP_PROP_POS_FRAMES, idxFrame);
                            if (!cap.read(frame)) { std::println(" end of file found"); break; }

                            std::print(" done, detecting chessboard...");
                            std::string baseName = p.path().stem().string();
                            std::string frameFilename = std::format("{}_{:05d}.jpg", baseName, idxFrame);
                            ChessboardImage ci(frame, chessboard, frameFilename);
                            std::println("{}", ci.isFound ? " found" : " not found");
                            if (ci.isFound)
                            {
                                chessboardImages.push_back(ci);
                                if (++kept >= target) {
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

void ChessboardData::drawCorners()
{
    for (auto & chessboardImage : chessboardImages)
        chessboardImage.drawCorners(chessboard);
}

void ChessboardData::drawBoxes(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
        chessboardImage.drawBox(chessboard, camera);
}

void ChessboardData::recoverPoses(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
        chessboardImage.recoverPose(chessboard, camera);
}

// ======================================================
// Camera
// ======================================================

void Camera::calibrate(ChessboardData & chessboardData)
{
    // 3D planar points
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    // Collected 2D points
    std::vector<std::vector<cv::Point2f>> rQOi_all;
    for (const auto & chessboardImage : chessboardData.chessboardImages)
        rQOi_all.push_back(chessboardImage.corners);
    assert(!rQOi_all.empty());

    // Image size
    imageSize = chessboardData.chessboardImages[0].image.size();

    // Prepare per-image object points
    std::vector<std::vector<cv::Point3f>> objectPoints(rQOi_all.size(), rPNn_all);

    // Outputs per image rotations/translations
    std::vector<cv::Mat> Thetacn_all, rNCc_all;
    double rms = 0.0;

    if (useFisheye()) {
        // ---- Fisheye calibration (equidistant model: k1..k4) ----
        flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
                cv::fisheye::CALIB_FIX_SKEW;

        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        distCoeffs   = cv::Mat::zeros(4, 1, CV_64F);

        std::print("Calibrating fisheye camera...");
        cv::TermCriteria term(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);
        rms = cv::fisheye::calibrate(
            objectPoints, rQOi_all, imageSize,
            cameraMatrix, distCoeffs,
            Thetacn_all, rNCc_all,
            flags, term
        );
        std::println(" done");
    } else {
        // ---- Classic pinhole calibration (your original model) ----
        flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL; // | cv::CALIB_TILTED_MODEL;
        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        distCoeffs   = cv::Mat::zeros(12, 1, CV_64F);

        std::print("Calibrating pinhole camera...");
        rms = cv::calibrateCamera(
            objectPoints, rQOi_all, imageSize,
            cameraMatrix, distCoeffs,
            Thetacn_all, rNCc_all,
            flags
        );
        std::println(" done");
    }

    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    // Set extrinsics onto each image
    assert(chessboardData.chessboardImages.size() == rNCc_all.size());
    assert(chessboardData.chessboardImages.size() == Thetacn_all.size());
    for (std::size_t k = 0; k < chessboardData.chessboardImages.size(); ++k)
    {
        Pose & Tnc = chessboardData.chessboardImages[k].Tnc;
        Pose Tcn(Thetacn_all[k], rNCc_all[k]); // world->camera
        Tnc = Tcn.inverse();                   // camera->world
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

static inline bool is_finite_vec3(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

void Camera::calcFieldOfView()
{
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    const double W = static_cast<double>(imageSize.width);
    const double H = static_cast<double>(imageSize.height);
    if (W <= 1.0 || H <= 1.0) { hFOV = vFOV = dFOV = 0.0; return; }

    const double cx = cameraMatrix.at<double>(0,2);
    const double cy = cameraMatrix.at<double>(1,2);

    auto angle_between = [](const cv::Vec3d& a, const cv::Vec3d& b) {
        double c = a.dot(b) / (cv::norm(a) * cv::norm(b));
        c = std::clamp(c, -1.0, 1.0);
        return std::acos(c);
    };

    // Center and border rays (computed via the model-specific pixel->ray)
    const cv::Vec3d uC  = pixelToVector(cv::Vec2d(cx,    cy   ));
    const cv::Vec3d uL  = pixelToVector(cv::Vec2d(0.0,   cy   ));
    const cv::Vec3d uR  = pixelToVector(cv::Vec2d(W-1.0, cy   ));
    const cv::Vec3d uT  = pixelToVector(cv::Vec2d(cx,    0.0  ));
    const cv::Vec3d uB  = pixelToVector(cv::Vec2d(cx,    H-1.0));
    const cv::Vec3d uTL = pixelToVector(cv::Vec2d(0.0,   0.0  ));
    const cv::Vec3d uBR = pixelToVector(cv::Vec2d(W-1.0, H-1.0));

    auto valid = [&](const cv::Vec3d& v){ return is_finite_vec3(v) && v[2] > 0.0; };

    // “Sum of center->edge angles” (same style as original)
    hFOV = (valid(uC) && valid(uL) && valid(uR)) ? angle_between(uC, uL) + angle_between(uC, uR) : 0.0;
    vFOV = (valid(uC) && valid(uT) && valid(uB)) ? angle_between(uC, uT) + angle_between(uC, uB) : 0.0;
    dFOV = (valid(uC) && valid(uTL) && valid(uBR)) ? angle_between(uC, uTL) + angle_between(uC, uBR) : 0.0;

    // --- azimuth-based theta limit table for FOV gating (unchanged idea) ---
    cosThetaLimit_.assign(360, -1.0);
    for (int deg = 0; deg < 360; ++deg) {
        double maxTheta = 0.0;
        double radAz = deg * CV_PI / 180.0;
        for (int rstep = 0; rstep < 2; ++rstep) {
            double r = 0.499 - 0.002 * rstep; // slightly inside
            double u = (W - 1) * (0.5 + r * std::cos(radAz));
            double v = (H - 1) * (0.5 + r * std::sin(radAz));
            cv::Vec3d ray = pixelToVector(cv::Vec2d(u, v));
            if (!valid(ray) || !valid(uC)) continue;
            double theta = angle_between(uC, ray);
            maxTheta = std::max(maxTheta, theta);
        }
        cosThetaLimit_[deg] = std::cos(maxTheta + (10.0 * CV_PI / 180.0)); // +10° margin
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
    Pose Tnc = bodyToCamera(Tnb); // Tnb*Tbc
    Pose Tcn = Tnc.inverse();     // world->camera
    cv::Vec3d rPCc = Tcn * rPNn;  // point in camera coords
    return rPCc / cv::norm(rPCc); // unit ray
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose & Tnb) const
{
    return vectorToPixel(worldToVector(rPNn, Tnb));
}

cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const
{
    cv::Point3d P(static_cast<double>(rPCc[0]/rPCc[2]),
                  static_cast<double>(rPCc[1]/rPCc[2]),
                  1.0);

    std::vector<cv::Point3d> obj{P};
    std::vector<cv::Point2d> img;

    cv::Mat rvec = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3,1,CV_64F);

    if (useFisheye()) {
        cv::fisheye::projectPoints(obj, img, rvec, tvec, cameraMatrix, distCoeffs);
    } else {
        cv::projectPoints(obj, rvec, tvec, cameraMatrix, distCoeffs, img);
    }

    return cv::Vec2d(img[0].x, img[0].y);
}

cv::Vec3d Camera::pixelToVector(const cv::Vec2d & rQOi) const
{
    if (useFisheye()) {
        std::vector<cv::Point2d> distorted{ cv::Point2d(rQOi[0], rQOi[1]) };
        std::vector<cv::Point2d> undistNorm; // normalized (P = I)
        cv::fisheye::undistortPoints(
            distorted, undistNorm, cameraMatrix, distCoeffs, cv::noArray(), cv::noArray()
        );
        cv::Vec3d rPCc(undistNorm[0].x, undistNorm[0].y, 1.0);
        return rPCc / cv::norm(rPCc);
    } else {
        std::vector<cv::Point2f> in, out;
        in.emplace_back(static_cast<float>(rQOi[0]), static_cast<float>(rQOi[1]));
        cv::undistortPoints(in, out, cameraMatrix, distCoeffs); // normalized
        cv::Vec3d rPCc(out[0].x, out[0].y, 1.0);
        return rPCc / cv::norm(rPCc);
    }
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    if (!std::isfinite(rPCc[0]) || !std::isfinite(rPCc[1]) || !std::isfinite(rPCc[2])) return false;
    if (rPCc[2] <= 1e-9) return false;

    cv::Vec3d dir = rPCc / cv::norm(rPCc);
    const cv::Vec3d z(0,0,1);

    double az = std::atan2(dir[1], dir[0]) * 180.0 / CV_PI; // deg
    int bin = static_cast<int>(std::lround(az));
    bin = (bin % 360 + 360) % 360; // wrap to [0,359]
    double cosang = dir.dot(z);
    if (bin >= 0 && bin < (int)cosThetaLimit_.size()) {
        if (cosang < cosThetaLimit_[bin])
            return false;
    }

    // Also ensure it maps to a valid pixel inside the image
    cv::Vec3d P(rPCc[0]/rPCc[2], rPCc[1]/rPCc[2], 1.0);
    cv::Vec2d px = vectorToPixel(P);
    if (!std::isfinite(px[0]) || !std::isfinite(px[1])) return false;

    return (px[0] >= 0.0 && px[0] < imageSize.width &&
            px[1] >= 0.0 && px[1] < imageSize.height);
}

bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose & Tnb) const
{
    return isVectorWithinFOV(worldToVector(rPNn, Tnb));
}

void Camera::write(cv::FileStorage & fs) const
{
    // Same structure/keys as original
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

    // Recompute FOV and limits
    calcFieldOfView();

    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.type() == CV_64F);
    assert(distCoeffs.cols == 1);
    assert(distCoeffs.type() == CV_64F);
}
