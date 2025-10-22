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
    MeasurementOutdoorFlowBundle(double time,
                                 const Camera & camera,
                                 const cv::Mat & imgk_raw,
                                 const cv::Mat & imgkm1_raw,
                                 const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1);

    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    // Helper functions for log likelihood and visualisation
    template <typename Scalar>
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predictFlowImpl(const Eigen::VectorX<Scalar> & x,
                                                             const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1,
                                                             const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const;

    template <typename Scalar>
    Scalar logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const;

    Eigen::Matrix<double, 2, Eigen::Dynamic> predictedFeatures(const Eigen::VectorXd & x,
                                                               const SystemEstimator & system) const;

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

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_;      // Measured features for previous frame (distorted pixels)
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_;        // Measured features for current frame  (distorted pixels)

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOikm1_;   // Undistorted features for previous frame (undistorted pixels or normalized per Camera API)
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_;     // Undistorted features for current frame

    std::vector<unsigned char> mask_;                       // Inlier mask

    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1_;         // Inlier undistorted homogeneous points in previous frame [ū v̄ 1]^T
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_;           // Inlier undistorted homogeneous points in current frame  [ū v̄ 1]^T

    double sigma_;                                          // Feature error stdev (pixels)
};

template <typename Scalar>
Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
MeasurementOutdoorFlowBundle::predictFlowImpl(const Eigen::VectorX<Scalar> & x,
                                              const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1,
                                              const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const
{
    (void)pk; // not used for the prediction branch decision (only pkm1 ray & pose are needed)

    assert(x.rows() >= 18);
    assert(x.cols() == 1);
    assert(pkm1.cols() == pk.cols());

    using Mat3 = Eigen::Matrix<Scalar,3,3>;
    using Vec3 = Eigen::Matrix<Scalar,3,1>;

    // State layout: [nu(6)=0..5, eta_k(6)=6..11, eta_km1(6)=12..17]
    const Eigen::Matrix<Scalar,6,1> etak   = x.template segment<6>(6);
    const Eigen::Matrix<Scalar,6,1> etakm1 = x.template segment<6>(12);

    // Poses and fixed Tbc
    const Pose<Scalar> Tnb_k   ( rpy2rot(etak  .template tail<3>()), etak  .template head<3>() );
    const Pose<Scalar> Tnb_km1 ( rpy2rot(etakm1.template tail<3>()), etakm1.template head<3>() );
    const Pose<Scalar> Tbc = camera_.Tbc;

    const Pose<Scalar> Tnc_k   = Tnb_k   * Tbc;  // world->cam at k
    const Pose<Scalar> Tnc_km1 = Tnb_km1 * Tbc;  // world->cam at k-1

    const Eigen::Matrix<Scalar,3,3> R_WCk   = Tnc_k  .rotationMatrix;
    const Eigen::Matrix<Scalar,3,1> r_WCk   = Tnc_k  .translationVector;
    const Eigen::Matrix<Scalar,3,3> R_WCkm1 = Tnc_km1.rotationMatrix;
    const Eigen::Matrix<Scalar,3,1> r_WCkm1 = Tnc_km1.translationVector;

    const Eigen::Matrix<Scalar,3,3> R_CkCkm1 = R_WCk.transpose() * R_WCkm1;

    // Build K^{-1} explicitly (avoids autodiff .inverse)
    const double fx = camera_.cameraMatrix.at<double>(0,0);
    const double fy = camera_.cameraMatrix.at<double>(1,1);
    const double cx = camera_.cameraMatrix.at<double>(0,2);
    const double cy = camera_.cameraMatrix.at<double>(1,2);

    Mat3 Kinv;
    Kinv << Scalar(1.0 / fx), Scalar(0),           Scalar(-cx / fx),
            Scalar(0),        Scalar(1.0 / fy),    Scalar(-cy / fy),
            Scalar(0),        Scalar(0),           Scalar(1);

    Eigen::Matrix<Scalar,3,Eigen::Dynamic> pk_hat(3, pkm1.cols());

    const double eps2 = 1e-12;   // guard for near-horizon

    for (int j = 0; j < pkm1.cols(); ++j)
    {
        // pkm1 holds UNDISTORTED pixels [ū, v̄, 1]; turn into a ray via K^{-1}
        const Vec3 pUbar_km1( Scalar(pkm1(0,j)), Scalar(pkm1(1,j)), Scalar(1) );
        const Vec3 dC_km1   = Kinv * pUbar_km1;           // direction in C_{k-1}
        const Vec3 dW_km1   = R_WCkm1 * dC_km1;           // direction in world

        // Branch in double to keep autodiff stable, then compute Scalar path
        const Scalar denom      = dW_km1.z();
        const double denom_val  = static_cast<double>(denom);
        const double denom2_val = denom_val * denom_val;
        const bool   nondeg     = (denom2_val > eps2);

        if (nondeg) {
            const Scalar s      = -r_WCkm1.z() / denom;
            const double s_val  = static_cast<double>(s);
            const bool   front  = (s_val > 0.0);
            if (front) {
                const Vec3 PW   = r_WCkm1 + s * dW_km1;
                const Vec3 PC_k = R_WCk.transpose() * (PW - r_WCk);
                pk_hat.col(j)   = PC_k;                   // homogeneous in C_k (z is depth)
                continue;
            }
        }

        // Sky/degenerate → plane at infinity: rotate direction only
        pk_hat.col(j) = R_CkCkm1 * dC_km1;
    }
    return pk_hat;
}

template <typename Scalar>
Scalar MeasurementOutdoorFlowBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const
{
    assert(pkm1_.cols() == pk_.cols());

    // Predict *undistorted* rays in C_k
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

#endif
