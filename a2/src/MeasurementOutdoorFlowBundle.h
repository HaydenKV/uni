#ifndef MEASUREMENTOUTDOORFLOWBUNDLE_H
#define MEASUREMENTOUTDOORFLOWBUNDLE_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "SystemEstimator.h"
#include "Pose.hpp"
#include "Camera.h"
#include "Measurement.h"

class MeasurementOutdoorFlowBundle : public Measurement
{
public:
    // Existing two-frame constructor (k vs k-1)
    MeasurementOutdoorFlowBundle(double time,
                                 const Camera & camera,
                                 const cv::Mat & imgk_raw,
                                 const cv::Mat & imgkm1_raw,
                                 const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1);

    // NEW: per-feature reference (k - n_j) + per-feature gaps
    // Keeps the downstream interface identical by storing rQ_ref as rQOikm1_.
    MeasurementOutdoorFlowBundle(double time,
                                 const Camera & camera,
                                 const cv::Mat & imgk_raw,
                                 const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik,   // current (k) DISTORTED pixels
                                 const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQ_ref,  // reference (k - n_j) DISTORTED pixels
                                 const std::vector<int> & gap_for_feat);                    // per-feature gaps

    // Measurement interface
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    // Helper functions for log likelihood and visualisation
    // NOTE: pkm1/pk carry **UNDISTORTED pixel homogeneous** points (ū, v̄, 1).
    template <typename Scalar>
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predictFlowImpl(const Eigen::VectorX<Scalar> & x,
                                                             const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1,
                                                             const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const;

    template <typename Scalar>
    Scalar logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const;

    // Predict **DISTORTED** pixels for drawing (blue predictions in your viewer)
    Eigen::Matrix<double, 2, Eigen::Dynamic> predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const;

    // Note: costOdometry is used only in Lab 11.
    template <typename Scalar>
    Scalar costOdometryImpl(const Eigen::VectorX<Scalar> & etak, const Eigen::VectorXd & etakm1) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g, Eigen::MatrixXd & H) const;

    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedPreviousFeatures() const;
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedCurrentFeatures() const;
    const std::vector<unsigned char> & inlierMask() const;

protected:
    const Camera & camera_;

    // Measured pixel features (DISTORTED pixels), shapes 2xN
    // rQOikm1_: reference frame (k-1 OR k - n_j)
    // rQOik_:   current frame (k)
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_;

    // Undistorted image points (normalized camera), shapes 2xN (not used directly in residual)
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOikm1_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_;

    std::vector<unsigned char> mask_;   // inlier mask (pre-inlier size N)

    // Inlier **UNDISTORTED pixel** homogeneous points, shapes 3xM
    // These are (ū, v̄, 1) in undistorted pixel units — consistent with K and K^{-1}.
    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1_;
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_;

    double sigma_;                       // feature error std (pixels, undistorted)

    // NEW:
    // gap_for_feat_: gaps aligned with input pairs (size N). Only set by per-feature ctor.
    // gap_inliers_:  gaps aligned with inlier-compact (size M). Used in predictFlowImpl.
    std::vector<int> gap_for_feat_;
    std::vector<int> gap_inliers_;
};

template <typename Scalar>
Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
MeasurementOutdoorFlowBundle::predictFlowImpl(const Eigen::VectorX<Scalar> & x,
                                              const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1,
                                              const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const
{
    assert(x.rows() >= 18);
    assert(x.cols() == 1);
    assert(pkm1.cols() == pk.cols());

    using Mat3 = Eigen::Matrix<Scalar,3,3>;
    using Vec3 = Eigen::Matrix<Scalar,3,1>;

    // State layout: [nu(6)=0..5, eta_k(6)=6..11, eta_km1(6)=12..17]
    const Eigen::Matrix<Scalar,6,1> etak   = x.template segment<6>(6);
    const Eigen::Matrix<Scalar,6,1> etakm1 = x.template segment<6>(12);

    // T^n_b(·) and fixed T^b_c
    const Pose<Scalar> Tnb_k   ( rpy2rot(etak  .template tail<3>()), etak  .template head<3>() );
    const Pose<Scalar> Tnb_km1 ( rpy2rot(etakm1.template tail<3>()), etakm1.template head<3>() );
    const Pose<Scalar> Tbc = camera_.Tbc;

    // Camera-at-k in world (R_WC, r_WC)
    const Pose<Scalar> Tnc_k = Tnb_k * Tbc;
    const Mat3 R_WCk = Tnc_k.rotationMatrix;
    const Vec3 r_WCk = Tnc_k.translationVector;

    // ΔT (world frame) from k-1 to k: Δ = Tnb_k * inv(Tnb_km1)
    const Mat3 Rdel = Tnb_k.rotationMatrix * Tnb_km1.rotationMatrix.transpose();
    const Vec3 tdel = Tnb_k.translationVector - Rdel * Tnb_km1.translationVector;

    // Find max n among inliers; if empty, assume 1
    int M = static_cast<int>(pkm1.cols());
    int nmax = 1;
    if (!gap_inliers_.empty()) {
        for (int gj : gap_inliers_) nmax = std::max(nmax, gj);
    }

    // Precompute Δ^n = (R^n, t^n) with t^n = sum_{i=0..n-1} R^i t
    std::vector<Mat3> Rpow(nmax+1);
    std::vector<Vec3> tpow(nmax+1);
    Rpow[0].setIdentity(); tpow[0].setZero();
    Mat3 Rp = Mat3::Identity();
    Vec3 tp = Vec3::Zero();
    for (int n = 1; n <= nmax; ++n) {
        tp += Rp * tdel;   // accumulate
        Rp  = Rp * Rdel;
        Rpow[n] = Rp;
        tpow[n] = tp;
    }

    // Intrinsics (build K^{-1})
    const double fx = camera_.cameraMatrix.at<double>(0,0);
    const double fy = camera_.cameraMatrix.at<double>(1,1);
    const double cx = camera_.cameraMatrix.at<double>(0,2);
    const double cy = camera_.cameraMatrix.at<double>(1,2);

    Mat3 Kinv;
    Kinv <<
        Scalar(1.0 / fx), Scalar(0),           Scalar(-cx / fx),
        Scalar(0),        Scalar(1.0 / fy),    Scalar(-cy / fy),
        Scalar(0),        Scalar(0),           Scalar(1);

    Eigen::Matrix<Scalar,3,Eigen::Dynamic> pk_hat(3, M);

    // Near-horizon guard band (~0.57°): uses world Z component of the ray
    const double tau = 0.010;        // radians
    const double eps2 = tau * tau;

    for (int j = 0; j < M; ++j)
    {
        // choose gap for this inlier j (default 1 if not supplied)
        const int gap = (!gap_inliers_.empty() && j < (int)gap_inliers_.size())
                        ? std::max(1, gap_inliers_[j]) : 1;

        // Tnb_{k-gap} = inv(Δ^gap) * Tnb_k
        const Mat3 Rkn = Rpow[gap].transpose();
        const Vec3 tkn = -Rkn * tpow[gap];

        const Mat3 Rnb_kn = Rkn * Tnb_k.rotationMatrix;
        const Vec3 rnb_kn = Rkn * Tnb_k.translationVector + tkn;

        const Pose<Scalar> Tnc_kn( Rnb_kn * Tbc.rotationMatrix,
                                   Rnb_kn * Tbc.translationVector + rnb_kn );

        const Mat3 R_WCkn = Tnc_kn.rotationMatrix;
        const Vec3 r_WCkn = Tnc_kn.translationVector;

        // UNDISTORTED *pixel* at reference -> ray in C_{k-n}
        const Vec3 pUbar_kn( Scalar(pkm1(0,j)), Scalar(pkm1(1,j)), Scalar(1) );
        const Vec3 dC_kn   = Kinv * pUbar_kn;       // in C_{k-n}
        const Vec3 dW_kn   = R_WCkn * dC_kn;        // in world

        const Scalar denom      = dW_kn.z();        // world Z
        const double denom_val  = static_cast<double>(denom);
        const double denom2_val = denom_val * denom_val;
        const bool   nondeg     = (denom2_val > eps2);

        if (nondeg) {
            // Intersect with ground Z=0: r.z + s * d.z = 0  -> s = -r.z/d.z
            const Scalar s      = -r_WCkn.z() / denom;
            const double s_val  = static_cast<double>(s);
            const bool   front  = (s_val > 0.0);
            if (front) {
                const Vec3 PW   = r_WCkn + s * dW_kn;                  // world
                const Vec3 PC_k = R_WCk.transpose() * (PW - r_WCk);    // camera k
                pk_hat.col(j)   = PC_k;
                continue;
            }
        }

        // Plane at infinity (rotate only)
        const Mat3 R_CkCkn = R_WCk.transpose() * R_WCkn;
        pk_hat.col(j) = R_CkCkn * dC_kn;
    }

    return pk_hat;
}

template <typename Scalar>
Scalar MeasurementOutdoorFlowBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const
{
    assert(pkm1_.cols() == pk_.cols());

    // Predict UNDISTORTED rays in C_k (homogeneous)
    Eigen::Matrix<Scalar,3,Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1_, pk_);

    // Dehomogenize to normalized image coords
    Eigen::Matrix<Scalar,2,Eigen::Dynamic> rNorm_hat(2, pk_hat.cols());
    const Scalar eps = Scalar(1e-9);
    for (int j = 0; j < pk_hat.cols(); ++j) {
        Scalar z = pk_hat(2,j);
        if (Eigen::numext::abs(z) < eps) z = (z >= Scalar(0)) ? eps : -eps;
        rNorm_hat(0,j) = pk_hat(0,j) / z;
        rNorm_hat(1,j) = pk_hat(1,j) / z;
    }

    // Map normalized -> UNDISTORTED *pixels*
    const double fx = camera_.cameraMatrix.at<double>(0,0);
    const double fy = camera_.cameraMatrix.at<double>(1,1);
    const double cx = camera_.cameraMatrix.at<double>(0,2);
    const double cy = camera_.cameraMatrix.at<double>(1,2);

    Eigen::Matrix<Scalar,2,Eigen::Dynamic> rUbar_hat(2, rNorm_hat.cols());
    rUbar_hat.row(0) = rNorm_hat.row(0).array() * Scalar(fx) + Scalar(cx);
    rUbar_hat.row(1) = rNorm_hat.row(1).array() * Scalar(fy) + Scalar(cy);

    // Measured UNDISTORTED pixels are pk_.topRows<2>()
    Eigen::Matrix<Scalar,2,Eigen::Dynamic> rUbar_meas = pk_.template topRows<2>().template cast<Scalar>();

    const Scalar inv_s2 = Scalar(1) / Scalar(sigma_ * sigma_);   // σ in pixels
    Scalar logLik = Scalar(0);
    for (int j = 0; j < rUbar_hat.cols(); ++j) {
        const Scalar dx = rUbar_hat(0,j) - rUbar_meas(0,j);
        const Scalar dy = rUbar_hat(1,j) - rUbar_meas(1,j);
        logLik -= Scalar(0.5) * (dx*dx + dy*dy) * inv_s2;
    }
    return logLik;
}

// Note: costOdometry is used only in Lab 11, not Assignment 2.
template <typename Scalar>
Scalar MeasurementOutdoorFlowBundle::costOdometryImpl(const Eigen::VectorX<Scalar> & etak, const Eigen::VectorXd & etakm1) const
{
    Eigen::VectorX<Scalar> x(18);
    x.setZero();
    x.template segment<6>(6)  = etak;
    x.template segment<6>(12) = etakm1;
    return -logLikelihoodImpl(x);
}

#endif // MEASUREMENTOUTDOORFLOWBUNDLE_H
