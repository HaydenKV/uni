#include <string>
#include <print>                  // C++23 std::print/std::println
#include <vector>
#include <array>
#include <algorithm>              // std::sort, std::min
#include <cmath>                  // std::sqrt
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Core>
#include "imagefeatures.h"

// ─────────────────────────────────────────────────────────────────────────────
// Harris corners
// Math: For each pixel, build the 2×2 second-moment matrix
//   M = Σ_{x,y∈W} w(x,y) [Ix^2  IxIy; IxIy  Iy^2]
// Score: R = det(M) − k·tr(M)^2, k∈[0.04,0.06]. We threshold on R and select
// top-N by score. Drawing is visualization-only; no change to detection logic.
// References: Harris–Stephens (1988); course notes §Cornerness via structure tensor.
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Grayscale image for gradient processing
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Harris response R (CV_32F)
    cv::Mat harrisResponse;
    int blockSize = 2;       // window size for M aggregation
    int apertureSize = 3;    // Sobel kernel for Ix,Iy
    double k = 0.04;         // empirical constant in R
    cv::cornerHarris(gray, harrisResponse, blockSize, apertureSize, k);

    // Use raw response for ranking; avoids scale distortion from normalization
    cv::Mat harrisNorm = harrisResponse.clone();

    struct Corner { cv::Point pt; float score; };
    std::vector<Corner> corners;

    // Coarse threshold on R; retain candidates
    float threshold = 0.001f;
    for (int y = 0; y < harrisNorm.rows; ++y) {
        for (int x = 0; x < harrisNorm.cols; ++x) {
            float response = harrisNorm.at<float>(y, x);
            if (response > threshold) {
                corners.push_back({cv::Point(x, y), response});
            }
        }
    }

    // Diagnostics (Task 1e style)
    std::println("Using harris feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", corners.size());

    // Rank by cornerness (R)
    std::sort(corners.begin(), corners.end(),
              [](const Corner& a, const Corner& b){ return a.score > b.score; });

    int count = std::min(maxNumFeatures, static_cast<int>(corners.size()));

    // Visualize all candidates (orange)
    for (const auto& c : corners) {
        cv::circle(imgout, c.pt, 4, cv::Scalar(0, 165, 255), -1);
    }
    // Highlight top-N (red ring + green index)
    for (int i = 0; i < count; ++i) {
        const auto& c = corners[i];
        std::println("idx: {} at point: ({},{}) Harris Score: {}", i, c.pt.x, c.pt.y, c.score);
        cv::circle(imgout, c.pt, 8, cv::Scalar(0, 0, 255), 1);
        cv::putText(imgout, std::to_string(i), c.pt + cv::Point(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

// ─────────────────────────────────────────────────────────────────────────────
/*
Shi–Tomasi corners (min-eigenvalue)
Math: same M as Harris. Score s = λ_min(M). Corners maximize λ_min (good
two-directional gradient energy). We threshold on s and rank top-N.
Reference: Shi & Tomasi (1994); course notes §Good Features to Track.
*/
cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Grayscale for gradient structure
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // λ_min(M) per pixel (CV_32F)
    cv::Mat minEigenResponse;
    int blockSize = 2;
    cv::cornerMinEigenVal(gray, minEigenResponse, blockSize);

    cv::Mat eigenNorm = minEigenResponse.clone();

    struct Corner { cv::Point pt; float score; };
    std::vector<Corner> corners;

    // Coarse threshold on λ_min
    float threshold = 0.01f;
    for (int y = 0; y < eigenNorm.rows; ++y) {
        for (int x = 0; x < eigenNorm.cols; ++x) {
            float response = eigenNorm.at<float>(y, x);
            if (response > threshold) {
                corners.push_back({cv::Point(x, y), response});
            }
        }
    }

    std::println("Using Shi-Tomasi feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", corners.size());

    std::sort(corners.begin(), corners.end(),
              [](const Corner& a, const Corner& b){ return a.score > b.score; });

    int count = std::min(maxNumFeatures, static_cast<int>(corners.size()));

    // Visualization: orange (all), red ring + label (top-N)
    for (const auto& c : corners) {
        cv::circle(imgout, c.pt, 3, cv::Scalar(0, 165, 255), -1);
    }
    for (int i = 0; i < count; ++i) {
        const auto& c = corners[i];
        std::println("idx: {} at point: ({},{}) Shi Score: {}", i, c.pt.x, c.pt.y, c.score);
        cv::circle(imgout, c.pt, 6, cv::Scalar(0, 0, 255), 2);
        cv::putText(imgout, std::to_string(i), c.pt + cv::Point(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

// ─────────────────────────────────────────────────────────────────────────────
/*
FAST keypoints
Math: Segment test on 16-circle: a pixel p is a corner if there exists a
contiguous arc of ≥ n pixels that are all > p+τ or < p−τ in intensity.
We apply OpenCV’s nonmax suppression on detector response and rank by response.
Reference: Rosten & Drummond (2006); course notes §FAST.
*/
cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Grayscale for intensity comparisons
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // FAST detection
    std::vector<cv::KeyPoint> keypoints;
    int threshold = 85;
    bool nonmaxSuppression = true;
    cv::Ptr<cv::FastFeatureDetector> detector =
        cv::FastFeatureDetector::create(threshold, nonmaxSuppression);
    detector->detect(gray, keypoints);

    std::println("Using FAST feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", keypoints.size());

    // Rank by response (contrast strength)
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b){ return a.response > b.response; });

    int count = std::min(maxNumFeatures, static_cast<int>(keypoints.size()));

    // Visualization
    for (const auto& kp : keypoints) {
        cv::circle(imgout, kp.pt, 3, cv::Scalar(0, 165, 255), -1);
    }
    for (int i = 0; i < count; ++i) {
        const auto& kp = keypoints[i];
        std::println("idx: {} at point: ({},{}) Score: {}", i, kp.pt.x, kp.pt.y, kp.response);
        cv::circle(imgout, kp.pt, 6, cv::Scalar(0, 0, 255), 2);
        cv::putText(imgout, std::to_string(i), kp.pt + cv::Point2f(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

// ─────────────────────────────────────────────────────────────────────────────
/*
ArUco detection (no pose)
Math: Dictionary-driven square fiducials; detection → quad extraction → ID
decoding. Outputs per-tag image-plane corners in OpenCV order (TL,TR,BR,BL).
Useful for association/visualization prior to pose.
*/
cv::Mat detectAndDrawArUco(const cv::Mat & img, int /*maxNumFeatures*/)
{
    cv::Mat imgout = img.clone();

    // Dictionary (6x6, 250 IDs)
    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::makePtr<cv::aruco::Dictionary>(
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)
        );

    // Detection containers
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;

    // Detect markers on the input image
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds);

    if (!markerIds.empty())
    {
        cv::aruco::drawDetectedMarkers(imgout, markerCorners, markerIds);

        std::println("Using aruco feature detector");
        std::println("Image width: {}", img.cols);
        std::println("Image height: {}", img.rows);
        std::println("Detected {} marker(s)", markerIds.size());

        // Sort by ID for stable, readable output
        std::vector<std::pair<int, std::vector<cv::Point2f>>> sortedMarkers;
        sortedMarkers.reserve(markerIds.size());
        for (size_t i = 0; i < markerIds.size(); ++i) {
            sortedMarkers.emplace_back(markerIds[i], markerCorners[i]);
        }
        std::sort(sortedMarkers.begin(), sortedMarkers.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });

        for (const auto& marker : sortedMarkers)
        {
            std::print("ID: {} with corners:", marker.first);
            for (int j = 0; j < 4; ++j) {
                const cv::Point2f pt = marker.second[j];
                std::print(" ({},{})", static_cast<int>(pt.x), static_cast<int>(pt.y));
            }
            std::print("\n");
        }
    }
    else
    {
        std::println("ArUco: No markers detected.");
    }

    return imgout;
}

// ─────────────────────────────────────────────────────────────────────────────
/*
ArUco + Pose via IPPE_SQUARE
Pipeline:
1) Detect markers with tuned cv::aruco::ArucoDetector (optionally draw rejected).
2) For each marker i, define object points in tag frame:
     X_j = (±L/2, ±L/2, 0), j ∈ {TL,TR,BR,BL} in OpenCV corner order.
3) Solve PnP: find (R_i, t_i) s.t. u_j ≈ π(K [R_i|t_i] X_j), using SOLVEPNP_IPPE_SQUARE
   (planar square, two-solution disambiguation).
4) Reprojection gate: ē = (1/4)Σ ||u_j − π(K [R_i|t_i] X_j)||_2; accept if ē ≤ threshold.
5) Draw frame axes of accepted poses for visual confirmation.

Notes:
- rvec is Rodrigues parameterization of R (axis-angle); tvec is translation (tag→camera).
- Camera model follows u ~ K [R|t] X, with radial/tangential distortion handled by distCoeffs.
*/
ArucoDetections detectArUcoPOSE(
    const cv::Mat& imgBGR,
    int dictionary,
    bool doCornerRefine,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    float tagSizeMeters,
    std::vector<cv::Vec3d>* outRvecs,
    std::vector<cv::Vec3d>* outTvecs,
    std::vector<double>* outMeanReprojErr,
    double reprojErrThreshPx,
    bool drawRejected
)
{
    CV_Assert(!imgBGR.empty());
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);
    CV_Assert(tagSizeMeters > 0.f);

    ArucoDetections out;
    out.annotated = imgBGR.clone();

    // Detector configuration (robust corner refinement + balanced thresholds)
    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(dictionary);
    cv::aruco::DetectorParameters params;

    // Corner refinement: subpixel for accuracy/stability
    params.cornerRefinementMethod = doCornerRefine
        ? cv::aruco::CORNER_REFINE_SUBPIX
        : cv::aruco::CORNER_REFINE_NONE;
    params.cornerRefinementWinSize       = 5;
    params.cornerRefinementMaxIterations = 100;
    params.cornerRefinementMinAccuracy   = 0.01;

    // Detection tuning: small tags near borders acceptable
    params.minMarkerPerimeterRate = 0.01f;
    params.maxMarkerPerimeterRate = 4.0f;

    params.adaptiveThreshWinSizeMin  = 3;
    params.adaptiveThreshWinSizeMax  = 23;
    params.adaptiveThreshWinSizeStep = 4;
    params.adaptiveThreshConstant    = 7;

    params.minCornerDistanceRate         = 0.04f;
    params.minDistanceToBorder           = 2;
    params.polygonalApproxAccuracyRate   = 0.05f;

    params.errorCorrectionRate = 0.6f;

    params.perspectiveRemovePixelPerCell         = 8;
    params.perspectiveRemoveIgnoredMarginPerCell = 0.13f;

    params.markerBorderBits = 1; // 6x6 dictionary default

    cv::aruco::ArucoDetector detector(dict, params);

    // Detect markers
    std::vector<int> ids_raw;
    std::vector<std::vector<cv::Point2f>> corners_raw, rejected;
    detector.detectMarkers(imgBGR, corners_raw, ids_raw, rejected);

    if (drawRejected && !rejected.empty()) {
        cv::aruco::drawDetectedMarkers(out.annotated, rejected, std::vector<int>(), cv::Scalar(100,100,100));
    }
    if (ids_raw.empty()) {
        return out; // nothing detected
    }

    // Visualize initial detections
    cv::aruco::drawDetectedMarkers(out.annotated, corners_raw, ids_raw);

    // Prepare PnP correspondences: tag corners in tag frame (meters)
    const float L = tagSizeMeters;
    const std::vector<cv::Point3f> objPts = {
        {-L/2.f,  L/2.f, 0.f},   // TL
        { L/2.f,  L/2.f, 0.f},   // TR
        { L/2.f, -L/2.f, 0.f},   // BR
        {-L/2.f, -L/2.f, 0.f}    // BL
    };

    if (outRvecs) outRvecs->clear();
    if (outTvecs) outTvecs->clear();
    if (outMeanReprojErr) outMeanReprojErr->clear();

    out.ids.reserve(ids_raw.size());
    out.corners.reserve(ids_raw.size());
    if (outRvecs) outRvecs->reserve(ids_raw.size());
    if (outTvecs) outTvecs->reserve(ids_raw.size());
    if (outMeanReprojErr) outMeanReprojErr->reserve(ids_raw.size());

    for (size_t i = 0; i < ids_raw.size(); ++i) {
        const std::vector<cv::Point2f>& imgPts = corners_raw[i];

        // Pose from planar square: IPPE_SQUARE is best-suited here
        cv::Vec3d rvec, tvec;
        bool ok = cv::solvePnP(
            objPts, imgPts,
            cameraMatrix, distCoeffs,
            rvec, tvec,
            false,                      // initial guess not used
            cv::SOLVEPNP_IPPE_SQUARE
        );
        if (!ok) continue;

        // Mean reprojection error gate (pixels)
        double meanErr = 0.0;
        {
            std::vector<cv::Point2f> reproj(4);
            cv::projectPoints(objPts, rvec, tvec, cameraMatrix, distCoeffs, reproj);
            for (int c = 0; c < 4; ++c) {
                const cv::Point2f d = reproj[c] - imgPts[c];
                meanErr += std::sqrt(d.dot(d));
            }
            meanErr *= 0.25; // average over 4 corners
        }
        if (reprojErrThreshPx > 0.0 && meanErr > reprojErrThreshPx) {
            continue; // reject low-quality pose
        }

        // Accept marker i
        out.ids.push_back(ids_raw[i]);

        std::array<cv::Point2f,4> c{};
        for (int k = 0; k < 4; ++k) c[k] = imgPts[k];
        out.corners.push_back(c);

        if (outRvecs) outRvecs->push_back(rvec);
        if (outTvecs) outTvecs->push_back(tvec);
        if (outMeanReprojErr) outMeanReprojErr->push_back(meanErr);

        // Visual confirmation: draw camera-frame axes at tag center
        cv::drawFrameAxes(out.annotated, cameraMatrix, distCoeffs, rvec, tvec, 0.4f * L, 2);
    }

    return out;
}
