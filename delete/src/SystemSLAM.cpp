#include <cstddef>
#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "rotation.hpp"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>


SystemSLAM::SystemSLAM(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{}


// Evaluate f(x) from the SDE dx = f(x)*dt + dw
/*
dynamics (no Jacobian):
Implements the deterministic part of (4)–(5):
  ṙ^n_{B/N} = R_nb(Θ) v^B_{N/B}
  Θ̇        = T(Θ) ω^B_{N/B}
  ṁ         = 0   (static landmarks)
Only f(x) is returned; process noise dw is handled in the filter via Q.
*/
Eigen::VectorXd SystemSLAM::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    assert(density.dim() == x.size());
    //
    //  dnu/dt =          0 + dwnu/dt
    // deta/dt = JK(eta)*nu +       0
    //   dm/dt =          0 +       0
    // \_____/   \________/   \_____/
    //  dx/dt  =    f(x)    +  dw/dt
    //
    //        [          0 ]
    // f(x) = [ JK(eta)*nu ]
    //        [          0 ] for all map states
    //
    //        [                    0 ]
    //        [                    0 ]
    // f(x) = [    Rnb(thetanb)*vBNb ]
    //        [ TK(thetanb)*omegaBNb ]
    //        [                    0 ] for all map states
    //
    Eigen::VectorXd f(x.size());
    f.setZero();

    // Extract velocity states ν = [vBNb; ωBNb]
    Eigen::Vector3d vBNb     = x.segment<3>(0);  // translational velocity (body)
    Eigen::Vector3d omegaBNb = x.segment<3>(3);  // angular velocity (body)

    // Pose subset η = [rBNn; Θ_nb]
    Eigen::Vector3d Thetanb  = x.segment<3>(9);  // RPY angles

    // R_nb(Θ) from Euler (assignment conventions)
    Eigen::Matrix3d Rnb = rpy2rot(Thetanb);

    // Position dynamics: ṙ = R_nb v (Eq. (4))
    f.segment<3>(6) = Rnb * vBNb;

    // Orientation dynamics: Θ̇ = T(Θ) ω (Eq. (4))
    const double phi = Thetanb(0);
    const double theta = Thetanb(1);

    Eigen::Matrix3d T;
    T << 1.0, std::sin(phi)*std::tan(theta),  std::cos(phi)*std::tan(theta),
         0.0, std::cos(phi),                 -std::sin(phi),
         0.0, std::sin(phi)/std::cos(theta),  std::cos(phi)/std::cos(theta);

    f.segment<3>(9) = T * omegaBNb;

    // Landmark dynamics: dm/dt = 0 (already zero)

    return f;
}


// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
/*
dynamics with Jacobian:
Returns f(x) and J = ∂f/∂x. By (4)–(5), J has structure
    [ 0     0           0 ]
J = [ d(Rv)/dν d(Rv)/dΘ  0 ]  with states ordered [ν(6), r(3), Θ(3), m(...)]
    [ 0     T(Θ)    d(Tω)/dΘ ]
Landmark columns are identically zero. We compute J wrt the first 12 states
via autodiff to ensure analytic consistency in tests.
*/
Eigen::VectorXd SystemSLAM::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    //
    //  Jacobian J = df/dx
    //    
    //     [  0                  0 0 ]
    // J = [ JK d(JK(eta)*nu)/deta 0 ]
    //     [  0                  0 0 ]
    //
    // where JK(eta) = blkdiag(Rnb(Theta), T(Theta))
    //

    assert(density.dim() == x.size());

    using autodiff::dual;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;
    using autodiff::val;

    const Eigen::Index nx = x.size();   // full state (includes landmarks)
    const Eigen::Index nVelPose = 12;   // [vBNb(3), ωBNb(3), rBNn(3), Θ(3)]

    // Differentiate only wrt [ν, η]; landmark columns are zero by model (5)
    Eigen::VectorX<dual> x_vel_pose_dual = x.head(nVelPose).cast<dual>();

    // f_dual builds full-dimension output but depends only on first 12 vars
    auto f_dual = [&](const Eigen::VectorX<dual>& xd) -> Eigen::VectorX<dual> {
        Eigen::VectorX<dual> fd(nx);
        fd.setZero();

        // Unpack ν, η from xd
        Eigen::Vector3<dual> vBNb     = xd.segment<3>(0);
        Eigen::Vector3<dual> omegaBNb = xd.segment<3>(3);
        Eigen::Vector3<dual> Thetanb  = xd.segment<3>(9);

        // R_nb(Θ) and T(Θ) (autodiff-safe)
        Eigen::Matrix3<dual> Rnb = rpy2rot(Thetanb);

        fd.segment<3>(6) = Rnb * vBNb; // ṙ = R v

        const dual phi = Thetanb(0);
        const dual theta = Thetanb(1);

        Eigen::Matrix3<dual> T;
        T << dual(1.0), sin(phi)*tan(theta),  cos(phi)*tan(theta),
             dual(0.0), cos(phi),             -sin(phi),
             dual(0.0), sin(phi)/cos(theta),  cos(phi)/cos(theta);

        fd.segment<3>(9) = T * omegaBNb; // Θ̇ = T ω

        return fd;
    };

    // J_vel_pose: (nx × 12); other columns are zero
    Eigen::VectorX<dual> f_dual_out;
    Eigen::MatrixXd J_vel_pose = jacobian(f_dual, wrt(x_vel_pose_dual), at(x_vel_pose_dual), f_dual_out);

    J.resize(nx, nx);
    J.setZero();
    J.leftCols(nVelPose) = J_vel_pose;

    // Optional sanity prints for a 12-dim (no-landmark) state block:
    if (x.size() == 12) {
        std::cout << "J(6,9) [∂ṙx/∂roll] = " << J(6,9) << std::endl;
        std::cout << "J(9,9) [∂φ̇/∂roll]  = " << J(9,9) << std::endl;

        double cross_deriv_norm = J.block<3,3>(6,9).norm();  // ∂ṙ/∂Θ
        std::cout << "||∂ṙ/∂Θ||_F        = " << cross_deriv_norm << std::endl;

        if (cross_deriv_norm < 1e-10) {
            std::cerr << "WARNING: Cross-derivatives near zero — check autodiff wiring.\n";
        }
    }

    // Extract f(x) values (strip dual → double)
    Eigen::VectorXd f(nx);
    for (Eigen::Index i = 0; i < nx; ++i) {
        f(i) = val(f_dual_out(i));
    }
    return f;
}

Eigen::VectorXd SystemSLAM::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

GaussianInfo<double> SystemSLAM::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    Eigen::MatrixXd SQ(6, 6);
    SQ.setZero();

    // Tunable spectral densities (These control how much the filter trusts the process model vs measurements)
    const double qv = 0.2;   // translational velocity drive (m·s^{-3/2})
    const double qw = 0.05;  // angular velocity drive (rad·s^{-3/2})
    
    // Diagonal noise (independent noise on each velocity component)
    SQ(0,0) = qv;  // x-velocity noise
    SQ(1,1) = qv;  // y-velocity noise
    SQ(2,2) = qv;  // z-velocity noise
    SQ(3,3) = 0.01;  // roll rate noise
    SQ(4,4) = 0.1;  // pitch rate noise
    SQ(5,5) = 1.0;  // yaw rate noise

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemSLAM::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    // Noise only affects velocity states (indices 0-5)
    std::vector<Eigen::Index> idxQ{0, 1, 2, 3, 4, 5};
    return idxQ;
}

cv::Mat & SystemSLAM::view()
{
    return view_;
};

const cv::Mat & SystemSLAM::view() const
{
    return view_;
};

GaussianInfo<double> SystemSLAM::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

GaussianInfo<double> SystemSLAM::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

GaussianInfo<double> SystemSLAM::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemSLAM::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

Eigen::Vector3d SystemSLAM::cameraPosition(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(camera, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

GaussianInfo<double> SystemSLAM::cameraPositionDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(camera, x, J); };
    return density.affineTransform(f);
}

Eigen::Vector3d SystemSLAM::cameraOrientationEuler(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(camera, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

GaussianInfo<double> SystemSLAM::cameraOrientationEulerDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(camera, x, J); };
    return density.affineTransform(f);    
}

GaussianInfo<double> SystemSLAM::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}
