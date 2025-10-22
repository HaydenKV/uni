#include <cstddef>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

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
        // TODO: Assignment 2
        return RK4SDEHelper(xdw, dt, J);
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
    // TODO: Implement in Assignment(s)

    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(t, x, u);

    // Jacobian J = df/dx
    J.resize(f.size(), x.size());
    J.setZero();
    // TODO: Implement in Assignment(s)

    return f;
}

Eigen::VectorXd SystemVisualNav::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

GaussianInfo<double> SystemVisualNav::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    Eigen::MatrixXd SQ;
    
    // TODO: Assignment(s)

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemVisualNav::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ;
    // TODO: Assignment(s)
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

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

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
