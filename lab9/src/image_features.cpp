#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "image_features.h"

PointFeature::PointFeature()
    : score(0)
    , x(0)
    , y(0)
{}

PointFeature::PointFeature(const double & score_, const double & x_, const double & y_)
    : score(score_)
    , x(x_)
    , y(y_)
{}

bool PointFeature::operator<(const PointFeature & other) const
{
    return (score > other.score);
}

// =====================================================================
// Lab-2 style detectors (feature-only, plotting/printing REMOVED) vvvvvvvvvvvvvvvvvvvvvv
// =====================================================================
// ---------------- Harris (parameterized) ----------------
static std::vector<PointFeature>
detectHarrisFeatures(const cv::Mat & img, int maxNumFeatures,
                     int blockSize, int aperture, double k, float respThresh)
{
    std::vector<PointFeature> feats;
    if (img.empty() || maxNumFeatures <= 0) return feats;

    // Grayscale
    cv::Mat gray;
    if (img.channels() == 1) gray = img;
    else cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Harris response (CV_32F)
    cv::Mat resp;
    cv::cornerHarris(gray, resp, blockSize, aperture, k);

    // Collect by threshold
    for (int v = 0; v < resp.rows; ++v) {
        const float* row = resp.ptr<float>(v);
        for (int u = 0; u < resp.cols; ++u) {
            float s = row[u];
            if (s > respThresh) {
                feats.emplace_back(static_cast<double>(s),
                                   static_cast<double>(u),
                                   static_cast<double>(v));
            }
        }
    }

    // Sort (desc by score) and cap
    std::sort(feats.begin(), feats.end());
    if ((int)feats.size() > maxNumFeatures) feats.resize(maxNumFeatures);
    return feats;
}

// Lab-2 default wrapper (unchanged behavior)
static std::vector<PointFeature>
detectHarrisFeatures(const cv::Mat & img, int maxNumFeatures)
{
    // Defaults from your Lab 2
    return detectHarrisFeatures(img, maxNumFeatures,
                                /*blockSize*/2, /*aperture*/3, /*k*/0.04,
                                /*respThresh*/0.001f);
}

// ---------------- Shi–Tomasi (parameterized) ----------------
static std::vector<PointFeature>
detectShiTomasiFeatures(const cv::Mat & img, int maxNumFeatures,
                        int blockSize, int ksize, float respThresh)
{
    std::vector<PointFeature> feats;
    if (img.empty() || maxNumFeatures <= 0) return feats;

    // Grayscale
    cv::Mat gray;
    if (img.channels() == 1) gray = img;
    else cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Min-eigenvalue response (CV_32F)
    cv::Mat resp;
    cv::cornerMinEigenVal(gray, resp, blockSize, ksize);

    // Collect by threshold
    for (int v = 0; v < resp.rows; ++v) {
        const float* row = resp.ptr<float>(v);
        for (int u = 0; u < resp.cols; ++u) {
            float s = row[u];
            if (s > respThresh) {
                feats.emplace_back(static_cast<double>(s),
                                   static_cast<double>(u),
                                   static_cast<double>(v));
            }
        }
    }

    // Sort (desc by score) and cap
    std::sort(feats.begin(), feats.end());
    if ((int)feats.size() > maxNumFeatures) feats.resize(maxNumFeatures);
    return feats;
}

// Lab-2 default wrapper (unchanged behavior)
static std::vector<PointFeature>
detectShiTomasiFeatures(const cv::Mat & img, int maxNumFeatures)
{
    // Defaults from your Lab 2
    return detectShiTomasiFeatures(img, maxNumFeatures,
                                   /*blockSize*/2, /*ksize*/3, /*respThresh*/0.01f);
}

// ---------------- FAST (parameterized) ----------------
static std::vector<PointFeature>
detectFASTFeatures(const cv::Mat & img, int maxNumFeatures,
                   int threshold, bool nonmax)
{
    std::vector<PointFeature> feats;
    if (img.empty() || maxNumFeatures <= 0) return feats;

    // Grayscale
    cv::Mat gray;
    if (img.channels() == 1) gray = img;
    else cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // FAST keypoints
    std::vector<cv::KeyPoint> kps;
    cv::Ptr<cv::FastFeatureDetector> det = cv::FastFeatureDetector::create(threshold, nonmax);
    det->detect(gray, kps);

    // Sort by response desc and convert to PointFeature
    std::sort(kps.begin(), kps.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b){ return a.response > b.response; });

    feats.reserve(std::min<int>(kps.size(), maxNumFeatures));
    for (const auto & kp : kps) {
        feats.emplace_back(static_cast<double>(kp.response),
                           static_cast<double>(kp.pt.x),
                           static_cast<double>(kp.pt.y));
        if ((int)feats.size() >= maxNumFeatures) break;
    }
    return feats;
}

// Lab-2 default wrapper (unchanged behavior)
static std::vector<PointFeature>
detectFASTFeatures(const cv::Mat & img, int maxNumFeatures)
{
    // Defaults from your Lab 2
    return detectFASTFeatures(img, maxNumFeatures,
                              /*threshold*/85, /*nonmax*/true);
}
// =====================================================================
// Lab-2 style detectors (feature-only, plotting/printing REMOVED) ^^^^^^^^^^^^^^
// =====================================================================

std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures)
{
    std::vector<PointFeature> features;
    // TODO: Lab 9
    // Choose a suitable feature detector
    // Save features above a certain texture threshold
    // Sort features by texture
    // Cap number of features to maxNumFeatures

    // ---------------- Harris (uncomment to use) ----------------
    // Tunables:
    // int   h_blockSize = 2;
    // int   h_aperture  = 3;
    // double h_k        = 0.04;
    // float  h_thresh   = 0.00002f;
    // features = detectHarrisFeatures(img, maxNumFeatures, h_blockSize, h_aperture, h_k, h_thresh);

    // ---------------- Shi–Tomasi (active by default) ----------------
    // Tunables:
    int   st_blockSize = 3;
    int   st_ksize     = 5;
    float st_thresh    = 0.01f;
    features = detectShiTomasiFeatures(img, maxNumFeatures, st_blockSize, st_ksize, st_thresh);

    // ---------------- FAST (uncomment to use) ----------------
    // Tunables:
    // int  f_threshold = 30;
    // bool f_nonmax    = true;
    // features = detectFASTFeatures(img, maxNumFeatures, f_threshold, f_nonmax);

    return features;
}
