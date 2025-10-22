#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <filesystem>
#include <ostream>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include "serialisation.hpp"

// -------- Model selector --------
enum class CameraModel { Auto, Fisheye, Pinhole };

// Helper class for working with elements of SE(3)
struct Pose
{
    cv::Matx33d rotationMatrix;     // R in SO(3)
    cv::Vec3d translationVector;    // r in R^3

    // Default constructor (R = I, r = 0)
    Pose();

    // Constructor from OpenCV rotation vector and translation vector
    Pose(const cv::Mat & rvec, const cv::Mat & tvec);

    // Group operation of SE(3)
    Pose operator*(const Pose & other) const;

    // Action of SE(3) on P3
    cv::Vec3d operator*(const cv::Vec3d & r) const;

    // Inverse element in SE(3)
    Pose inverse() const;
};

struct Chessboard
{
    cv::Size boardSize;
    float squareSize;

    void write(cv::FileStorage & fs) const;                 // OpenCV serialisation
    void read(const cv::FileNode & node);                   // OpenCV serialisation

    std::vector<cv::Point3f> gridPoints() const;
    friend std::ostream & operator<<(std::ostream &, const Chessboard &);
};

struct Camera;

struct ChessboardImage
{
    ChessboardImage(const cv::Mat &, const Chessboard &, const std::filesystem::path & = "");
    cv::Mat image;
    std::filesystem::path filename;
    Pose Tnc;                                               // Extrinsic camera parameters
    std::vector<cv::Point2f> corners;                       // Chessboard corners in image [rQOi]
    bool isFound;
    void drawCorners(const Chessboard &);
    void drawBox(const Chessboard &, const Camera &);
    void recoverPose(const Chessboard &, const Camera &);
};

struct ChessboardData
{
    explicit ChessboardData(const std::filesystem::path &); // Load from config file

    Chessboard chessboard;
    std::vector<ChessboardImage> chessboardImages;

    void drawCorners();
    void drawBoxes(const Camera &);
    void recoverPoses(const Camera &);
};

struct Camera
{
    void calibrate(ChessboardData &);                       // Calibrate camera (fisheye or pinhole per model)
    void printCalibration() const;

    Pose cameraToBody(const Pose & Tnc) const;              // Convert camera pose Tnc to body pose Tnb
    Pose bodyToCamera(const Pose & Tnb) const;              // Convert body pose Tnb to camera pose Tnc
    cv::Vec3d worldToVector(const cv::Vec3d & rPNn, const Pose & Tnb) const;
    cv::Vec2d worldToPixel(const cv::Vec3d &, const Pose &) const;
    cv::Vec2d vectorToPixel(const cv::Vec3d &) const;
    cv::Vec3d pixelToVector(const cv::Vec2d &) const;

    bool isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose & Tnb) const;
    bool isVectorWithinFOV(const cv::Vec3d & rPCc) const;

    void calcFieldOfView();
    void write(cv::FileStorage &) const;                    // OpenCV serialisation (unchanged format)
    void read(const cv::FileNode &);                        // OpenCV serialisation

    // ---- Settings ----
    CameraModel model = CameraModel::Auto;                  // Set this in main: Fisheye / Pinhole / Auto

    // ---- Intrinsics/Extrinsics ----
    cv::Mat cameraMatrix;                                   // Camera matrix
    cv::Mat distCoeffs;                                     // Distortion coefficients
    int flags = 0;                                          // Calibration flags
    cv::Size imageSize;                                     // Image size
    Pose Tbc;                                               // Relative pose of camera in body coordinates (Rbc, rCBb)

    // ---- FOVs (radians); same names as original ----
    double hFOV = 0.0;                                      // Horizontal field of view
    double vFOV = 0.0;                                      // Vertical field of view
    double dFOV = 0.0;                                      // Diagonal field of view

    std::vector<double> cosThetaLimit_;                     // For FOV gating

    // Helper: decide fisheye usage considering `model`
    bool useFisheye() const {
        if (model == CameraModel::Fisheye) return true;
        if (model == CameraModel::Pinhole) return false;
        // Auto: infer from coeff count (4 -> fisheye)
        return !distCoeffs.empty() && distCoeffs.total() == 4;
    }
};

#endif
