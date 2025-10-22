#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <filesystem>
#include <Eigen/Core>
#include <array>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include "serialisation.hpp"
#include "Pose.hpp"

namespace CamDefaults {
    inline int BorderMarginPx = 15;   // for the conserative functions in assignment
}

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
    Posed Tnc;                                               // Extrinsic camera parameters
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
    cv::Vec3d worldToVector(const cv::Vec3d & rPNn, const Posed & Tnb) const;
    cv::Vec2d worldToPixel(const cv::Vec3d &, const Posed &) const;
    cv::Vec2d vectorToPixel(const cv::Vec3d &) const;
    template <typename Scalar> Eigen::Vector2<Scalar> vectorToPixel(const Eigen::Vector3<Scalar> &) const;
    Eigen::Vector2d vectorToPixel(const Eigen::Vector3d &, Eigen::Matrix23d &) const;

    cv::Vec3d pixelToVector(const cv::Vec2d &) const;

    bool isWorldWithinFOV(const cv::Vec3d & rPNn, const Posed & Tnb) const;
    bool isVectorWithinFOV(const cv::Vec3d & rPCc) const;

    Eigen::Matrix<double, 2, Eigen::Dynamic> undistort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOi) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic> undistort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQOi) const;
    Eigen::Matrix<double, 2, Eigen::Dynamic> distort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQbarOi) const;
    Eigen::Matrix<double, 3, Eigen::Dynamic> distort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQbarOi) const;

    void calcFieldOfView();
    void write(cv::FileStorage &) const;                    // OpenCV serialisation
    void read(const cv::FileNode &);                        // OpenCV serialisation

    // Conservative, pixel-space bounds checks (default = CamDefaults::BorderMarginPx)
    bool isPixelInside(const cv::Point2f& uv, int margin = -1) const;
    bool isPixelInside(const cv::Point2d& uv, int margin = -1) const;
    bool isPixelInside(const Eigen::Vector2d& uv, int margin = -1) const;

    // Convenience for ArUco-style 4-corner checks
    bool areCornersInside(const std::array<cv::Point2f,4>& corners, int margin = -1) const;
    bool areCornersInside(const Eigen::Matrix<double,8,1>& uv8, int margin = -1) const;

    // FOV + pixel-margin gate for a 3D ray in camera coords
    bool isVectorWithinFOVConservative(const cv::Vec3d& rPCc, int margin = -1) const;

    cv::Mat cameraMatrix;                                   // Camera matrix
    cv::Mat distCoeffs;                                     // Lens distortion coefficients
    int flags = 0;                                          // Calibration flags
    cv::Size imageSize;                                     // Image size

    Posed Tbc;                                       // Relative pose of camera in body coordinates (Rbc, rCBb)

    std::vector<double> cosThetaLimit_;

private:
    double hFOV = 0.0;                                      // Horizonal field of view
    double vFOV = 0.0;                                      // Vertical field of view
    double dFOV = 0.0;                                      // Diagonal field of view
};

#include <cmath>
#include <stdexcept>
#include <opencv2/calib3d.hpp>

/*
Template projection π(K,dist):
Given r^c_{P/C} = (X,Y,Z), compute normalised (x,y) = (X/Z,Y/Z),
apply rational + thin-prism distortion, then map to pixels
[u,v]^T = [fx*xd + cx, fy*yd + cy]^T. Used by autodiff paths in measurement models.
*/
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
    // iii) Auto diff ONLY ----------------------------------------------------------------
    // Normalised image coords
    const Scalar X = rPCc(0), Y = rPCc(1), Z = rPCc(2);
    const Scalar invZ = Scalar(1) / Z;
    const Scalar x = X * invZ;
    const Scalar y = Y * invZ;

    const Scalar r2 = x*x + y*y;
    const Scalar r4 = r2*r2;
    const Scalar r6 = r4*r2;

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

    return rQOi;
}



#endif

