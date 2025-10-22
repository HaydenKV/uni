#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H

#include <vector>              // std::vector for ArucoDetections
#include <array>               // std::array for 4-corner storage
#include <opencv2/core.hpp>
#include <Eigen/Core>

// Harris: R = det(M) − k·tr(M)^2, with M = Σ w·[Ix Iy]^T[Ix Iy]
// Returns BGR image with all detected corners (orange), top-N highlighted.
cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures);

// Shi–Tomasi: score = λ_min(M), λ_min is the smallest eigenvalue of M
// Returns BGR image with all detected corners (orange), top-N highlighted.
cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures);

// FAST: segment test on a Bresenham circle; nonmax suppression applied
// Returns BGR image with all detected keypoints (orange), top-N highlighted.
cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures);

// ArUco: dictionary-based square fiducials; draws boxes/IDs for all detections
cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures);

// Grouped ArUco outputs for downstream SLAM/visualization
struct ArucoDetections
{
    std::vector<int> ids;                                // Marker IDs
    std::vector<std::array<cv::Point2f,4>> corners;      // TL,TR,BR,BL per marker (image px)
    cv::Mat annotated;                                   // Input image with overlays
};

// ArUco + pose (IPPE_SQUARE):
// Solves PnP for each tag using corner correspondences
//   u ~ π(K [R|t] X), reprojection error e = (1/4)Σ||u_meas - u_proj||_2
// Rejects detections with mean reprojection error > reprojErrThreshPx.
// Optionally returns {rvec,tvec} (Rodrigues) and mean reprojection error per tag.
ArucoDetections detectArUcoPOSE(
    const cv::Mat& imgBGR,
    int dictionary,
    bool doCornerRefine,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    float tagSizeMeters,
    std::vector<cv::Vec3d>* outRvecs = nullptr,
    std::vector<cv::Vec3d>* outTvecs = nullptr,
    std::vector<double>* outMeanReprojErr = nullptr,
    double reprojErrThreshPx = 4.0,
    bool drawRejected = false
);

#endif
