#include <cstddef>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"
#include "rotation.hpp"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

SystemVisualNav::SystemVisualNav(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{

}

SystemVisualNav * SystemVisualNav::clone() const
{
    return new SystemVisualNav(*this);
}

void SystemVisualNav::predict(double time)
{
    double dt = time - time_;
    assert(dt >= 0);
    if (dt == 0.0) return;

    // Augment state density with independent noise increment dw ~ N^{-1}(0, LambdaQ/dt)
    // [ x] ~ N^{-1}([ eta ], [ Lambda,          0 ])
    // [dw]         ([   0 ]  [      0, LambdaQ/dt ])

    auto pdw = processNoiseDensity(dt); // p(dw(idxQ)[k])
    auto pxdw = density*pdw;            // p(x[k], dw(idxQ)[k]) = p(x[k])*p(dw(idxQ)[k])

    // Phi maps [ x[k]; dw(idxQ)[k] ] to x[k+1]
    auto Phi = [&](const Eigen::VectorXd & xdw, Eigen::MatrixXd & J)
    {
        // Use the generic RK4 SDE integrator to propagate the state+noise.
        // Then apply the stochastic-clone rule: set ζ_next := η_old (clone previous pose)
        // and fix the Jacobian rows corresponding to the clone so they reflect
        // ∂ζ_next/∂η_old = I and zeros elsewhere.

        // xdw = [ x_old; dw ] where x_old is full state at time k
        const std::size_t nq = processNoiseIndex().size();
        const std::size_t nx = xdw.size() - nq;

        // store old pose η (indices 6..11)
        Eigen::VectorXd x_old = xdw.head(nx);
        Eigen::VectorXd eta_old = x_old.segment<6>(6);

        // call RK4 helper to get nominal next state and Jacobian
        Eigen::VectorXd xnext = RK4SDEHelper(xdw, dt, J);

        // If state doesn't have clone block, nothing more to do
        if (nx >= 18)
        {
            // overwrite ζ_next (indices 12..17) with η_old
            xnext.segment<6>(12) = eta_old;

            // J is (nx) x (nx + nq) with first nx columns = ∂xnext/∂xold
            // and last nq columns = ∂xnext/∂dw. We must set clone rows to depend
            // on η_old (cols 6..11) and not on old-ζ or dw.
            // Zero the clone rows first
            J.block(12, 0, 6, nx).setZero();
            if (nq > 0)
                J.block(12, nx, 6, nq).setZero();

            // Set ∂ζ_next / ∂η_old = I6
            J.block(12, 6, 6, 6) = Eigen::Matrix<double,6,6>::Identity();
        }

        return xnext;
    };
    
    // Map p(x[k], dw(idxQ)[k]) to p(x[k+1])
    density = pxdw.affineTransform(Phi);

    time_ = time;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    assert(density.dim() == x.size());

    Eigen::VectorXd f(x.size());
    f.setZero();
    // State ordering: [ν(6), η(6), ζ(6), m...]
    // ν = [v_b; ω_b] (indices 0..5)
    // η = [r_B/N^n; Θ_b^n] (indices 6..11)
    // ζ clone indices 12..17 (no continuous dynamics)
    // landmarks static

    const Eigen::Index nx = x.size();

    // Extract body-fixed velocities
    Eigen::Vector3d vB = x.segment<3>(0);
    Eigen::Vector3d omegaB = x.segment<3>(3);

    // Pose angles
    Eigen::Vector3d Thetanb = x.segment<3>(9);

    // Rotation R_nb(Θ)
    Eigen::Matrix3d Rnb = rpy2rot(Thetanb);

    // Position dynamics: ṙ = R_nb * vB
    f.segment<3>(6) = Rnb * vB;

    // Orientation dynamics: Θ̇ = T(Θ) * ω_b
    const double phi = Thetanb(0);
    const double theta = Thetanb(1);

    Eigen::Matrix3d T;
    T << 1.0, std::sin(phi)*std::tan(theta),  std::cos(phi)*std::tan(theta),
         0.0, std::cos(phi),                 -std::sin(phi),
         0.0, std::sin(phi)/std::cos(theta),  std::cos(phi)/std::cos(theta);

    f.segment<3>(9) = T * omegaB;

    // ν dynamics: translational acceleration is driven by process noise (deterministic part zero)
    // angular velocity deterministic drift is zero

    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    using autodiff::dual;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;
    using autodiff::val;

    const Eigen::Index nx = x.size();
    const Eigen::Index nVelPose = 12; // [v(3), w(3), r(3), Theta(3)]

    // Differentiate only wrt first 12 states; clone and landmarks columns are zero
    Eigen::VectorX<dual> x12_dual = x.head(nVelPose).cast<dual>();

    auto f_dual = [&](const Eigen::VectorX<dual> & xd) -> Eigen::VectorX<dual> {
        Eigen::VectorX<dual> fd(nx);
        fd.setZero();

        Eigen::Vector3<dual> vB = xd.segment<3>(0);
        Eigen::Vector3<dual> omegaB = xd.segment<3>(3);
        Eigen::Vector3<dual> Thetanb = xd.segment<3>(9);

        Eigen::Matrix3<dual> Rnb = rpy2rot(Thetanb);
        fd.segment<3>(6) = Rnb * vB;

        const dual phi = Thetanb(0);
        const dual theta = Thetanb(1);

        Eigen::Matrix3<dual> T;
        T << dual(1.0), sin(phi)*tan(theta),  cos(phi)*tan(theta),
             dual(0.0), cos(phi),             -sin(phi),
             dual(0.0), sin(phi)/cos(theta),  cos(phi)/cos(theta);

        fd.segment<3>(9) = T * omegaB;

        return fd;
    };

    Eigen::VectorX<dual> f_dual_out;
    Eigen::MatrixXd J_vel_pose = jacobian(f_dual, wrt(x12_dual), at(x12_dual), f_dual_out);

    J.resize(nx, nx);
    J.setZero();
    J.leftCols(nVelPose) = J_vel_pose;

    // Extract f values
    // Eigen::VectorXd f(nx);
    // for (Eigen::Index i = 0; i < nx; ++i) {
    //     f(i) = val(f_dual_out(i));
    // }
    Eigen::VectorXd f = dynamics(t, x, u);
    return f;
}

Eigen::VectorXd SystemVisualNav::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

GaussianInfo<double> SystemVisualNav::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    // Only translational body acceleration is driven by white noise -> 3 components
    Eigen::MatrixXd SQ(3,3);
    SQ.setZero();

    // Tunable spectral densities (match SystemSLAM's style)
    const double qv = 0.1; // translational accel drive (m·s^{-3/2})

    SQ(0,0) = qv;
    SQ(1,1) = qv;
    SQ(2,2) = qv;

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemVisualNav::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected into translational velocity components (indices 0,1,2)
    std::vector<Eigen::Index> idxQ{0, 1, 2};
    return idxQ;
}

cv::Mat & SystemVisualNav::view()
{
    return view_;
};

const cv::Mat & SystemVisualNav::view() const
{
    return view_;
};

std::size_t SystemVisualNav::numberLandmarks() const
{
    return (density.dim() - 18)/3;
}

std::size_t SystemVisualNav::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 18 + 3*idxLandmark;    
}

GaussianInfo<double> SystemVisualNav::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

GaussianInfo<double> SystemVisualNav::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

GaussianInfo<double> SystemVisualNav::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemVisualNav::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

Eigen::Vector3d SystemVisualNav::cameraPosition(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(camera, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

GaussianInfo<double> SystemVisualNav::cameraPositionDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(camera, x, J); };
    return density.affineTransform(f);
}

Eigen::Vector3d SystemVisualNav::cameraOrientationEuler(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(camera, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

GaussianInfo<double> SystemVisualNav::cameraOrientationEulerDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(camera, x, J); };
    return density.affineTransform(f);    
}

GaussianInfo<double> SystemVisualNav::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}
