#include <cstddef>
#include <print>
#include <numeric>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include "SystemEstimator.h"
#include "MeasurementOutdoorFlowBundle.h"

MeasurementOutdoorFlowBundle::MeasurementOutdoorFlowBundle(double time,
                                                           const Camera & camera,
                                                           const cv::Mat & imgk_raw,
                                                           const cv::Mat & imgkm1_raw,
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
    , sigma_(2.0) // pixels
{
    // (Assignment note) updateMethod_ can be selected by caller; not used in Lab 11.

    // --- Lab 11 parameters ---
    const int divisor               = 2;     // LK on reduced scale; plotting also uses this
    const int maxNumFeatures        = 1000;
    const int minNumFeatures        = 800;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,0.01);
    cv::Size subPixWinSize(11, 11);
    cv::Size winSize(21, 21);

    // Convert to grayscale
    cv::Mat imgk_gray, imgkm1_gray;
    cv::cvtColor(imgk_raw,  imgk_gray,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgkm1_raw,imgkm1_gray,cv::COLOR_BGR2GRAY);

    // Scale images
    cv::Mat imgk_scaled, imgkm1_scaled;
    cv::resize(imgk_gray,   imgk_scaled,   cv::Size(), 1.0/divisor, 1.0/divisor);
    cv::resize(imgkm1_gray, imgkm1_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);

    // (A) Seed features on k-1 image (scaled) if we have too few
    std::vector<cv::Point2f> rQOikm1_scaled;
    if (rQOikm1_.cols() < minNumFeatures)
    {
        cv::goodFeaturesToTrack(
            imgkm1_scaled, rQOikm1_scaled,
            maxNumFeatures, 0.01, 7.0, cv::noArray(), 3, false, 0.04
        );
        if (!rQOikm1_scaled.empty())
            cv::cornerSubPix(imgkm1_scaled, rQOikm1_scaled, subPixWinSize, cv::Size(-1,-1), termcrit);
    }
    else
    {
        rQOikm1_scaled.resize(rQOikm1_.cols());
        for (int j = 0; j < rQOikm1_.cols(); ++j)
        {
            rQOikm1_scaled[j].x = (float)(rQOikm1_(0, j)/divisor);
            rQOikm1_scaled[j].y = (float)(rQOikm1_(1, j)/divisor);
        }
    }

    // (B) Track k-1 -> k with pyramidal LK on *scaled* frames
    std::vector<cv::Point2f> rQOik_scaled;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        imgkm1_scaled, imgk_scaled,
        rQOikm1_scaled, rQOik_scaled,
        status, err, winSize, /*maxLevel=*/3, termcrit,
        /*flags=*/0, /*minEigThreshold=*/1e-4f
    );

    // (C) Keep only good, in-bounds matches (scaled coordinates)
    std::vector<cv::Point2f> rQOikm1_kept, rQOik_kept;
    rQOikm1_kept.reserve(rQOik_scaled.size());
    rQOik_kept.reserve(rQOik_scaled.size());
    for (size_t j = 0; j < rQOik_scaled.size(); ++j)
    {
        if (!status[j]) continue;
        const cv::Point2f &p = rQOik_scaled[j];
        if (p.x < 0 || p.y < 0 || p.x >= imgk_scaled.cols || p.y >= imgk_scaled.rows) continue;
        rQOikm1_kept.push_back(rQOikm1_scaled[j]);
        rQOik_kept.push_back(p);
    }
    int np = (int)rQOik_kept.size();
    std::println("After filtering by status, there are {} associations.", np);

    // (D) Unscale back to original pixel coords and store measured points
    rQOik_.resize(2, np);
    rQOikm1_.resize(2, np);
    for (int j = 0; j < np; ++j)
    {
        rQOik_(0, j)   = rQOik_kept[j].x      * divisor;
        rQOik_(1, j)   = rQOik_kept[j].y      * divisor;

        rQOikm1_(0, j) = rQOikm1_kept[j].x    * divisor;
        rQOikm1_(1, j) = rQOikm1_kept[j].y    * divisor;
    }

    // (E) Undistort to normalized image coords or undistorted pixels (per Camera API)
    rQbarOik_.resize(2, np);
    rQbarOikm1_.resize(2, np);
    {
        Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi_k(2, np), rQOi_km1(2, np);
        rQOi_k   = rQOik_;
        rQOi_km1 = rQOikm1_;
        rQbarOik_   = camera_.undistort(rQOi_k);
        rQbarOikm1_ = camera_.undistort(rQOi_km1);
    }

    // (F) RANSAC fundamental matrix on UNDISTORTED points → inlier mask
    std::vector<cv::Point2f> rQbarQOik(np), rQbarQOikm1(np);
    for (int j = 0; j < np; ++j) {
        // Use scaled coords for RANSAC pixel threshold consistency
        rQbarQOik[j]   = cv::Point2f((float)(rQbarOik_(0,j)   / divisor),
                                     (float)(rQbarOik_(1,j)   / divisor));
        rQbarQOikm1[j] = cv::Point2f((float)(rQbarOikm1_(0,j) / divisor),
                                     (float)(rQbarOikm1_(1,j) / divisor));
    }
    cv::Mat inlierMask;
    const double ransacPixThresh = 1.0, confidence = 0.999;
    (void)cv::findFundamentalMat(rQbarQOik, rQbarQOikm1, cv::FM_RANSAC, ransacPixThresh, confidence, inlierMask);

    mask_.assign(np, 0);
    if (!inlierMask.empty()) {
        for (int j = 0; j < np; ++j) mask_[j] = inlierMask.at<uchar>(j);
    }
    int nInliers = std::count(mask_.begin(), mask_.end(), (unsigned char)1);
    std::println("No. inliers = {}, No. outliers  = {}", nInliers, np - nInliers);

    // (G) Build *inlier* undistorted homogeneous points p[k-1], p[k]
    pk_   = Eigen::MatrixXd::Ones(3, nInliers);
    pkm1_ = Eigen::MatrixXd::Ones(3, nInliers);
    for (int j = 0, ii = 0; j < np; ++j) if (mask_[j]) {
        pkm1_.col(ii).head<2>() = rQbarOikm1_.col(j);   // undistorted pixels (or normalized; must be consistent with header math)
        pk_.col(ii).head<2>()   = rQbarOik_.col(j);
        ++ii;
    }
}

Eigen::VectorXd MeasurementOutdoorFlowBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y;
    throw std::runtime_error("Not implemented");
    return y;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> & MeasurementOutdoorFlowBundle::trackedPreviousFeatures() const
{
    return rQOikm1_;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> & MeasurementOutdoorFlowBundle::trackedCurrentFeatures() const
{
    return rQOik_;
}

const std::vector<unsigned char> & MeasurementOutdoorFlowBundle::inlierMask() const
{
    return mask_;
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    return logLikelihoodImpl(x);
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    g.resize(x.size());
    g.setZero();
    // (Assignment 2) Fill if needed; Lab 11 uses costOdometry directly.
    return logLikelihood(x, system);
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();
    // (Assignment 2) Fill if needed; Lab 11 uses costOdometry directly.
    return logLikelihood(x, system, g);
}

Eigen::Matrix<double, 2, Eigen::Dynamic>
MeasurementOutdoorFlowBundle::predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    (void)system;

    std::size_t np = rQOik_.cols();

    // Predict undistorted homogeneous image points in current frame
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk(3, np);
    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1(3, np);
    pkm1.topRows<2>() = rQbarOikm1_;
    pkm1.row(2).setOnes();
    pk.topRows<2>() = rQbarOik_;
    pk.row(2).setOnes();

    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1, pk);
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_hat = pk_hat.topRows<2>().array().rowwise()/pk_hat.row(2).array();
    assert(rQbarOik_hat.cols() == (int)np);

    // Convert UNDISTORTED normalized → UNDISTORTED pixels
    const double fx = camera_.cameraMatrix.at<double>(0,0);
    const double fy = camera_.cameraMatrix.at<double>(1,1);
    const double cx = camera_.cameraMatrix.at<double>(0,2);
    const double cy = camera_.cameraMatrix.at<double>(1,2);
    Eigen::Matrix<double, 2, Eigen::Dynamic> rUbar(2, (int)np);
    rUbar.row(0) = rQbarOik_hat.row(0).array() * fx + cx;
    rUbar.row(1) = rQbarOik_hat.row(1).array() * fy + cy;

    // Camera::distort expects undistorted *pixels* and returns distorted *pixels*
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
