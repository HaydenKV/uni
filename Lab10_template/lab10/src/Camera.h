#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <filesystem>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include "serialisation.hpp"
#include "Pose.hpp"

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
    Pose<double> Tnc;                                               // Extrinsic camera parameters
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

namespace Eigen {
using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}

struct Camera
{
    void calibrate(ChessboardData &);                       // Calibrate camera from chessboard data
    void printCalibration() const;

    template <typename Scalar> Pose<Scalar> cameraToBody(const Pose<Scalar> & Tnc) const { return Tnc*Tbc.inverse(); } // Tnb = Tnc*Tcb
    template <typename Scalar> Pose<Scalar> bodyToCamera(const Pose<Scalar> & Tnb) const { return Tnb*Tbc; } // Tnc = Tnb*Tbc
    cv::Vec3d worldToVector(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const;
    cv::Vec2d worldToPixel(const cv::Vec3d &, const Pose<double> &) const;
    cv::Vec2d vectorToPixel(const cv::Vec3d &) const;
    template <typename Scalar> Eigen::Vector2<Scalar> vectorToPixel(const Eigen::Vector3<Scalar> &) const;
    Eigen::Vector2d vectorToPixel(const Eigen::Vector3d &, Eigen::Matrix23d &) const;

    cv::Vec3d pixelToVector(const cv::Vec2d &) const;

    bool isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const;
    bool isVectorWithinFOV(const cv::Vec3d & rPCc) const;

    Eigen::Matrix<double, 2, Eigen::Dynamic> undistort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOi) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic> undistort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQOi) const;
    Eigen::Matrix<double, 2, Eigen::Dynamic> distort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQbarOi) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic> distort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQbarOi) const;

    void calcFieldOfView();
    void write(cv::FileStorage &) const;                    // OpenCV serialisation
    void read(const cv::FileNode &);                        // OpenCV serialisation

    cv::Mat cameraMatrix;                                   // Camera matrix
    cv::Mat distCoeffs;                                     // Lens distortion coefficients
    int flags = 0;                                          // Calibration flags
    cv::Size imageSize;                                     // Image size

    Pose<double> Tbc;                                       // Relative pose of camera in body coordinates (Rbc, rCBb)

private:
    double hFOV = 0.0;                                      // Horizonal field of view
    double vFOV = 0.0;                                      // Vertical field of view
    double dFOV = 0.0;                                      // Diagonal field of view
};

#include <cmath>
#include <stdexcept>
#include <opencv2/calib3d.hpp>

template <typename Scalar>
Eigen::Vector2<Scalar> Camera::vectorToPixel(const Eigen::Vector3<Scalar> & rPCc) const
{
    bool isRationalModel    = (flags & cv::CALIB_RATIONAL_MODEL)    == cv::CALIB_RATIONAL_MODEL;
    bool isThinPrismModel   = (flags & cv::CALIB_THIN_PRISM_MODEL)  == cv::CALIB_THIN_PRISM_MODEL;
    bool isTiltedModel      = (flags & cv::CALIB_TILTED_MODEL)      == cv::CALIB_TILTED_MODEL;

    if (isTiltedModel)
    {
        throw std::logic_error("Tilted camera model not currently implemented.");
    }

    // Camera matrix parameters
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    // Lens distortion coefficients
    // theta = k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, taux, tauy]]]] has 4, 5, 8, 12 or 14 elements

    // Radial and tangential distortion coefficients
    double k1 = distCoeffs.at<double>(0, 0);
    double k2 = distCoeffs.at<double>(1, 0);
    double p1 = distCoeffs.at<double>(2, 0);
    double p2 = distCoeffs.at<double>(3, 0);
    double k3 = distCoeffs.at<double>(4, 0);

    // Rational distortion coefficients
    double k4, k5, k6;
    if (isRationalModel)
    {
        k4 = distCoeffs.at<double>(5, 0);
        k5 = distCoeffs.at<double>(6, 0);
        k6 = distCoeffs.at<double>(7, 0);
    }
    else
    {
        k4 = k5 = k6 = 0.0;
    }

    // Thin prism distortion coefficients
    double s1, s2, s3, s4;
    if (isThinPrismModel)
    {
        s1 = distCoeffs.at<double>(8, 0);
        s2 = distCoeffs.at<double>(9, 0);
        s3 = distCoeffs.at<double>(10, 0);
        s4 = distCoeffs.at<double>(11, 0);
    }
    else
    {
        s1 = s2 = s3 = s4 = 0.0;
    }

    Eigen::Vector2<Scalar> rQOi;
    // TODO: Lab 8 (optional)
    return rQOi;
}

#endif

