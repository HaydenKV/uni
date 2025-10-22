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

    void calcFieldOfView();
    void write(cv::FileStorage &) const;                    // OpenCV serialisation
    void read(const cv::FileNode &);                        // OpenCV serialisation

    cv::Mat cameraMatrix;                                   // Camera matrix
    cv::Mat distCoeffs;                                     // Lens distortion coefficients
    int flags = 0;                                          // Calibration flags
    cv::Size imageSize;                                     // Image size

    Pose<double> Tbc;                                       // Relative pose of camera in body coordinates (Rbc, rCBb)

    std::vector<double> cosThetaLimit_;

private:
    double hFOV = 0.0;                                      // Horizonal field of view
    double vFOV = 0.0;                                      // Vertical field of view
    double dFOV = 0.0;                                      // Diagonal field of view
};

#include <cmath>
#include <opencv2/calib3d.hpp>

template <typename Scalar>
Eigen::Vector2<Scalar> Camera::vectorToPixel(const Eigen::Vector3<Scalar> & rPCc) const
{
    bool isRationalModel    = (flags & cv::CALIB_RATIONAL_MODEL) == cv::CALIB_RATIONAL_MODEL;
    bool isThinPrismModel   = (flags & cv::CALIB_THIN_PRISM_MODEL) == cv::CALIB_THIN_PRISM_MODEL;
    assert(isRationalModel && isThinPrismModel);

    Eigen::Vector2<Scalar> rQOi;
    // TODO: Lab 8 (optional)
    // iii) Auto diff ONLY ----------------------------------------------------------------
    // Normalised image coords
    const Scalar X = rPCc(0), Y = rPCc(1), Z = rPCc(2);
    const Scalar invZ = Scalar(1) / Z;
    const Scalar x = X * invZ;
    const Scalar y = Y * invZ;

    const Scalar r2 = x*x + y*y;
    const Scalar r4 = r2*r2;
    const Scalar r6 = r4*r2;

    // Intrinsics (double -> Scalar)
    const Scalar fx = static_cast<Scalar>(cameraMatrix.at<double>(0,0));
    const Scalar fy = static_cast<Scalar>(cameraMatrix.at<double>(1,1));
    const Scalar cx = static_cast<Scalar>(cameraMatrix.at<double>(0,2));
    const Scalar cy = static_cast<Scalar>(cameraMatrix.at<double>(1,2));

    // Distortion coeff access
    auto dc = [&](int i)->Scalar {
        if (i < distCoeffs.rows * distCoeffs.cols) return static_cast<Scalar>(distCoeffs.at<double>(i,0));
        return Scalar(0);
    };
    const Scalar k1 = dc(0),  k2 = dc(1),  p1 = dc(2),  p2 = dc(3),  k3 = dc(4);
    const Scalar k4 = dc(5),  k5 = dc(6),  k6 = dc(7),  s1 = dc(8),  s2 = dc(9);
    const Scalar s3 = dc(10), s4 = dc(11);
    // If you later enable TILTED_MODEL with 14 coeffs, add tauX,tauY here.

    // Rational radial
    const Scalar num   = Scalar(1) + k1*r2 + k2*r4 + k3*r6;
    const Scalar den   = Scalar(1) + k4*r2 + k5*r4 + k6*r6;
    const Scalar cdist = num / den;

    // Tangential
    const Scalar x_tan = Scalar(2)*p1*x*y + p2*(r2 + Scalar(2)*x*x);
    const Scalar y_tan = p1*(r2 + Scalar(2)*y*y) + Scalar(2)*p2*x*y;

    // Thin-prism
    const Scalar x_pr  = s1*r2 + s2*r4;
    const Scalar y_pr  = s3*r2 + s4*r4;

    // Distorted normalised
    const Scalar xd = x*cdist + x_tan + x_pr;
    const Scalar yd = y*cdist + y_tan + y_pr;

    // Pixels
    rQOi << fx*xd + cx, fy*yd + cy;
    // iii) Auto diff ONLY ----------------------------------------------------------------
    return rQOi;
}

#endif

