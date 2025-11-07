#include <cstddef>
#include <print>
#include <numeric>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>

#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>

#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include "SystemEstimator.h"
#include "MeasurementOutdoorFlowBundle.h"

// --------------------------------------------------------------------------------------
// Existing two-frame constructor (k vs k-1)
// --------------------------------------------------------------------------------------
MeasurementOutdoorFlowBundle::MeasurementOutdoorFlowBundle(
    double time, const Camera & camera,
    const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw,
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1)
    : Measurement(time)
    , camera_(camera)
    , rQOikm1_(rQOikm1)
    , rQOik_()
    , rQbarOikm1_()
    , rQbarOik_()
    , mask_()
    , pkm1_()
    , pk_()
    , sigma_(1.2)   // tighter helps avoid slow pitch "explaining" tiny flows
    , gap_for_feat_()
    , gap_inliers_()
{
    // LK / reseed / gating runs on scaled images for speed/viewer parity
    const int divisor        = 2;     // image scaling factor
    const int maxNumFeatures = 1400;  // reseed max
    const int minNumFeatures = 1200;  // reseed threshold

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.01);
    cv::Size subPixWinSize(11, 11);
    cv::Size winSize(21, 21);

    // Convert to grayscale
    cv::Mat imgk_gray, imgkm1_gray;
    cv::cvtColor(imgk_raw,  imgk_gray,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgkm1_raw,imgkm1_gray,cv::COLOR_BGR2GRAY);

    // Scale
    cv::Mat imgk_scaled, imgkm1_scaled;
    cv::resize(imgk_gray,  imgk_scaled,  cv::Size(), 1.0/divisor, 1.0/divisor);
    cv::resize(imgkm1_gray,imgkm1_scaled,cv::Size(), 1.0/divisor, 1.0/divisor);

    // (A) Reseed (k-1) if too few features were passed in
    std::vector<cv::Point2f> rQOikm1_scaled;
    if (rQOikm1_.cols() < minNumFeatures)
    {
        // Ground-prior mask on scaled previous image
        cv::Mat mask(imgkm1_scaled.size(), CV_8UC1, cv::Scalar(0));
        const int rows = imgkm1_scaled.rows;
        const int cols = imgkm1_scaled.cols;
        const int y0   = static_cast<int>(rows * 0.45);
        const int border = 8;
        cv::Rect roi(border, y0, cols - 2*border, rows - y0 - border);
        roi &= cv::Rect(0, 0, cols, rows);
        mask(roi).setTo(255);

        cv::goodFeaturesToTrack(imgkm1_scaled, rQOikm1_scaled,
                                maxNumFeatures, 0.01, 8.0,
                                mask, 3, false, 0.04);
        if (!rQOikm1_scaled.empty()) {
            cv::cornerSubPix(imgkm1_scaled, rQOikm1_scaled, subPixWinSize, cv::Size(-1,-1), termcrit);
        }
    }
    else
    {
        rQOikm1_scaled.resize(rQOikm1_.cols());
        for (int j = 0; j < rQOikm1_.cols(); ++j) {
            rQOikm1_scaled[j].x = static_cast<float>(rQOikm1_(0, j) / divisor);
            rQOikm1_scaled[j].y = static_cast<float>(rQOikm1_(1, j) / divisor);
        }
    }

    // (B) Pyramidal LK: track (k-1) -> (k) on scaled frames
    std::vector<cv::Point2f> rQOik_scaled;
    std::vector<unsigned char> status, status_fb;
    std::vector<float> err, err_fb;

    cv::calcOpticalFlowPyrLK(imgkm1_scaled, imgk_scaled,
                             rQOikm1_scaled, rQOik_scaled,
                             status, err,
                             winSize, 3, termcrit,
                             0, 1e-4f);

    // forward-backward check
    std::vector<cv::Point2f> rQOikm1_back;
    cv::calcOpticalFlowPyrLK(imgk_scaled, imgkm1_scaled,
                             rQOik_scaled, rQOikm1_back,
                             status_fb, err_fb,
                             winSize, 3, termcrit, 0, 1e-4f);

    const float fb_thresh = 1.5f; // scaled px
    for (size_t j = 0; j < rQOik_scaled.size(); ++j) {
        if (!status[j] || !status_fb[j]) { status[j] = 0; continue; }
        const cv::Point2f d = rQOikm1_back[j] - rQOikm1_scaled[j];
        if (d.dot(d) > fb_thresh*fb_thresh) status[j] = 0;
    }

    // (C) Keep only good, in-bounds matches (on scaled images)
    std::vector<cv::Point2f> rQOikm1_kept, rQOik_kept;
    rQOikm1_kept.reserve(rQOik_scaled.size());
    rQOik_kept.reserve(rQOik_scaled.size());

    for (size_t j = 0; j < rQOik_scaled.size(); ++j)
    {
        if (!status[j]) continue;
        const cv::Point2f &p = rQOik_scaled[j];

        // reject too small / too large motion (scaled px)
        const float dx = p.x - rQOikm1_scaled[j].x;
        const float dy = p.y - rQOikm1_scaled[j].y;
        const float flow = std::sqrt(dx*dx + dy*dy);
        if (flow < 0.15f || flow > 40.0f) continue;

        if (p.x < 0 || p.y < 0 || p.x >= imgk_scaled.cols || p.y >= imgk_scaled.rows) continue;

        rQOikm1_kept.push_back(rQOikm1_scaled[j]);
        rQOik_kept .push_back(p);
    }

    const int np = static_cast<int>(rQOik_kept.size());
    std::println("After filtering by status, there are {} associations.", np);

    // (D) Unscale kept matches back to original pixel coords
    rQOik_.resize(2, np);
    rQOikm1_.resize(2, np);
    for (int j = 0; j < np; ++j)
    {
        rQOik_(0, j)   = rQOik_kept[j].x   * divisor;
        rQOik_(1, j)   = rQOik_kept[j].y   * divisor;
        rQOikm1_(0, j) = rQOikm1_kept[j].x * divisor;
        rQOikm1_(1, j) = rQOikm1_kept[j].y * divisor;
    }

    // (E) Undistort to normalized image coords (Eigen 2×N)
    rQbarOik_.resize(2, np);
    rQbarOikm1_.resize(2, np);
    {
        Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi_k  (2, np),
                                                 rQOi_km1(2, np);
        for (int j = 0; j < np; ++j) {
            rQOi_k  (0,j) = rQOik_  (0,j);
            rQOi_k  (1,j) = rQOik_  (1,j);
            rQOi_km1(0,j) = rQOikm1_(0,j);
            rQOi_km1(1,j) = rQOikm1_(1,j);
        }
        rQbarOik_   = camera_.undistort(rQOi_k);
        rQbarOikm1_ = camera_.undistort(rQOi_km1);
    }

    // (F) RANSAC fundamental matrix on UNDISTORTED points -> inlier mask
    std::vector<cv::Point2f> rQbarQOik(np), rQbarQOikm1(np);
    for (int j = 0; j < np; ++j) {
        rQbarQOik[j]   = cv::Point2f(static_cast<float>(rQbarOik_(0,j)   / divisor),
                                     static_cast<float>(rQbarOik_(1,j)   / divisor));
        rQbarQOikm1[j] = cv::Point2f(static_cast<float>(rQbarOikm1_(0,j) / divisor),
                                     static_cast<float>(rQbarOikm1_(1,j) / divisor));
    }
    cv::Mat inlierMask;
    const double ransacPixThresh = 0.6, confidence = 0.999;
    cv::findFundamentalMat(rQbarQOik, rQbarQOikm1, cv::FM_RANSAC, ransacPixThresh, confidence, inlierMask);

    mask_.assign(np, 0);
    if (!inlierMask.empty()) {
        for (int j = 0; j < np; ++j) mask_[j] = inlierMask.at<uchar>(j);
    }
    int nInliers = std::count(mask_.begin(), mask_.end(), static_cast<unsigned char>(1));
    std::println("No. inliers = {}, No. outliers  = {}", nInliers, np - nInliers);

    // (G) Build inlier UNDISTORTED homogeneous points + gaps (all ones here)
    pk_   = Eigen::MatrixXd::Ones(3, nInliers);
    pkm1_ = Eigen::MatrixXd::Ones(3, nInliers);
    gap_inliers_.clear(); gap_inliers_.reserve(nInliers);

    for (int j = 0, ii = 0; j < np; ++j) if (mask_[j]) {
        pkm1_.col(ii).head<2>() = rQbarOikm1_.col(j);
        pk_  .col(ii).head<2>() = rQbarOik_  .col(j);
        gap_inliers_.push_back(1);   // two-frame path uses gap=1
        ++ii;
    }
}

// --------------------------------------------------------------------------------------
// NEW overload: per-feature reference rQ_ref (at k - n_j) and per-feature gaps
// Stores rQ_ref as rQOikm1_ so downstream math is unchanged.
// --------------------------------------------------------------------------------------
MeasurementOutdoorFlowBundle::MeasurementOutdoorFlowBundle(
    double time, const Camera & camera,
    const cv::Mat & imgk_raw,
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik,   // current (k)
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQ_ref,  // reference (k - n_j)
    const std::vector<int> & gap_for_feat)                    // per-feature gaps
    : Measurement(time)
    , camera_(camera)
    , rQOikm1_(rQ_ref)    // store ref as "previous"
    , rQOik_(rQOik)
    , rQbarOikm1_()
    , rQbarOik_()
    , mask_()
    , pkm1_()
    , pk_()
    , sigma_(1.2)
    , gap_for_feat_(gap_for_feat)
    , gap_inliers_()
{
    (void)imgk_raw; // not needed here, but kept to mirror other constructor signature
    const int divisor = 2;

    const int np = static_cast<int>(rQOik_.cols());
    if (np == 0) {
        return;
    }

    // Undistort to normalized (2xN)
    rQbarOikm1_.resize(2, np);
    rQbarOik_.resize(2, np);
    rQbarOik_   = camera_.undistort(rQOik_);
    rQbarOikm1_ = camera_.undistort(rQOikm1_);

    // -------- Per-gap RANSAC --------
    // Build groups by gap value
    std::unordered_map<int, std::vector<int>> groupIdx;
    groupIdx.reserve(8);
    for (int j = 0; j < np; ++j) {
        int g = 1;
        if (j < (int)gap_for_feat_.size())
            g = std::max(1, gap_for_feat_[j]);
        groupIdx[g].push_back(j);
    }

    mask_.assign(np, 0);

    const double ransacPixThresh = 0.6, confidence = 0.999;

    for (auto &kv : groupIdx)
    {
        const std::vector<int> &idxs = kv.second;
        if ((int)idxs.size() < 8) {
            // too small for RANSAC; accept as-is
            for (int j : idxs) mask_[j] = 1;
            continue;
        }

        std::vector<cv::Point2f> A, B;
        A.reserve(idxs.size());
        B.reserve(idxs.size());
        for (int j : idxs) {
            A.emplace_back(static_cast<float>(rQbarOik_(0,j)   / divisor),
                           static_cast<float>(rQbarOik_(1,j)   / divisor));
            B.emplace_back(static_cast<float>(rQbarOikm1_(0,j) / divisor),
                           static_cast<float>(rQbarOikm1_(1,j) / divisor));
        }

        cv::Mat m;
        cv::Mat inl;
        m = cv::findFundamentalMat(A, B, cv::FM_RANSAC, ransacPixThresh, confidence, inl);

        if (!inl.empty()) {
            for (int k = 0; k < inl.rows; ++k) {
                if (inl.at<uchar>(k)) {
                    const int j_global = idxs[k];
                    mask_[j_global] = 1;
                }
            }
        } else {
            // if RANSAC failed, accept all in the group (safe fallback)
            for (int j : idxs) mask_[j] = 1;
        }
    }

    const int nInliers = std::count(mask_.begin(), mask_.end(), static_cast<unsigned char>(1));
    std::println("No. inliers = {}, No. outliers  = {}", nInliers, np - nInliers);

    // Build inlier UNDISTORTED homogeneous points + inlier-aligned gaps
    pk_   = Eigen::MatrixXd::Ones(3, nInliers);
    pkm1_ = Eigen::MatrixXd::Ones(3, nInliers);
    gap_inliers_.clear(); gap_inliers_.reserve(nInliers);

    for (int j = 0, ii = 0; j < np; ++j) if (mask_[j]) {
        pkm1_.col(ii).head<2>() = rQbarOikm1_.col(j); // ref
        pk_  .col(ii).head<2>() = rQbarOik_  .col(j); // curr
        int g = 1;
        if (j < (int)gap_for_feat_.size()) g = std::max(1, gap_for_feat_[j]);
        gap_inliers_.push_back(g);
        ++ii;
    }
}

// --------------------------------------------------------------------------------------
// Boilerplate Measurement interface
// --------------------------------------------------------------------------------------
Eigen::VectorXd MeasurementOutdoorFlowBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y;
    throw std::runtime_error("Not implemented");
    return y;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> &
MeasurementOutdoorFlowBundle::trackedPreviousFeatures() const
{
    return rQOikm1_;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> &
MeasurementOutdoorFlowBundle::trackedCurrentFeatures() const
{
    return rQOik_;
}

const std::vector<unsigned char> &
MeasurementOutdoorFlowBundle::inlierMask() const
{
    return mask_;
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & /*system*/) const
{
    return logLikelihoodImpl(x);
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & /*system*/, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    g.resize(x.size());
    g.setZero();
    return logLikelihood(x, /*system*/ *(const SystemEstimator*)nullptr); // delegates to impl
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & /*system*/, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();
    return logLikelihood(x, /*system*/ *(const SystemEstimator*)nullptr, g);
}

Eigen::Matrix<double, 2, Eigen::Dynamic>
MeasurementOutdoorFlowBundle::predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & /*system*/) const
{
    const std::size_t np = rQOik_.cols();

    // Predict UNDISTORTED homogeneous image points in current frame
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk(3, np);
    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1(3, np);
    pkm1.topRows<2>() = rQbarOikm1_;
    pkm1.row(2).setOnes();
    pk.topRows<2>()   = rQbarOik_;
    pk.row(2).setOnes();

    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1, pk);
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_hat = pk_hat.topRows<2>().array().rowwise() / pk_hat.row(2).array();
    assert(rQbarOik_hat.cols() == static_cast<int>(np));

    // Compute image coordinates (with lens distortion)
    const double fx = camera_.cameraMatrix.at<double>(0, 0);
    const double fy = camera_.cameraMatrix.at<double>(1, 1);
    const double cx = camera_.cameraMatrix.at<double>(0, 2);
    const double cy = camera_.cameraMatrix.at<double>(1, 2);

    Eigen::Matrix<double, 2, Eigen::Dynamic> rUbar(2, np);
    rUbar.row(0) = rQbarOik_hat.row(0).array() * fx + cx;
    rUbar.row(1) = rQbarOik_hat.row(1).array() * fy + cy;

    // Camera::distort expects UNDISTORTED PIXELS and returns DISTORTED PIXELS
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = camera_.distort(rUbar);
    return rQOik_hat;
}

// Note: costOdometry is used only in Lab 11, not Assignment 2.
double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1) const
{
    return costOdometryImpl(etak, etakm1);
}

double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g) const
{
    // Forward-mode autodifferentiation
    Eigen::Matrix<autodiff::dual, Eigen::Dynamic, 1> etakdual = etak.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(&MeasurementOutdoorFlowBundle::costOdometryImpl<autodiff::dual>, wrt(etakdual), at(this, etakdual, etakm1), fdual);
    return val(fdual);
}

double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Forward-mode autodifferentiation
    Eigen::Matrix<autodiff::dual2nd, Eigen::Dynamic, 1> etakdual = etak.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = hessian(&MeasurementOutdoorFlowBundle::costOdometryImpl<autodiff::dual2nd>, wrt(etakdual), at(this, etakdual, etakm1), fdual, g);
    return val(fdual);
}
