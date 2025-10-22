#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "GaussianInfo.hpp"
#include "MeasurementGaussianLikelihood.h"
#include "MeasurementRADAR.h"

const double MeasurementRADAR::r1 = 5000;    // Horizontal position of sensor [m]
const double MeasurementRADAR::r2 = 5000;    // Vertical position of sensor [m]

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y)
    : MeasurementGaussianLikelihood(time, y)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // updateMethod_ = UpdateMethod::AFFINE;
    // updateMethod_ = UpdateMethod::GAUSSNEWTON;
    // updateMethod_ = UpdateMethod::LEVELBERGMARQUARDT;
}

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y, int verbosity)
    : MeasurementGaussianLikelihood(time, y, verbosity)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // updateMethod_ = UpdateMethod::AFFINE;
    // updateMethod_ = UpdateMethod::GAUSSNEWTON;
    // updateMethod_ = UpdateMethod::LEVELBERGMARQUARDT;
}

MeasurementRADAR::~MeasurementRADAR() = default;

std::string MeasurementRADAR::getProcessString() const
{
    return "RADAR measurement update:";
}

// Evaluate h(x) from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    // Eigen::VectorXd h(1);
    // // TODO
    // return h;

    // State: x = [h, v, beta]
    const double h = x(0);

    // Range from radar at (r1, r2) to target at (0, h)
    const double range = std::sqrt(r1*r1 + (r2 - h)*(r2 - h));

    Eigen::VectorXd yhat(1);
    yhat(0) = range;
    return yhat;
}

// Evaluate h(x) and its Jacobian J = dh/fx from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & dhdx) const
{
    Eigen::VectorXd h = predict(x, system);

    //              dh_i
    // dhdx(i, j) = ----
    //              dx_j
    dhdx.resize(h.size(), x.size());
    dhdx.setZero();
    // TODO: Set non-zero elements of dhdx
    const double r     = h(0);
    const double hgt   = x(0);
    const double dr_dh = (hgt - r2) / r;

    dhdx(0, 0) = dr_dh;

    return h;
}

Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & dhdx, Eigen::Tensor<double, 3> & d2hdx2) const
{
    Eigen::VectorXd h = predict(x, system, dhdx);

    //                    d^2 h_i     d 
    // d2hdx2(i, j, k) = --------- = ---- dhdx(i, j)
    //                   dx_j dx_k   dx_k
    d2hdx2.resize(h.size(), x.size(), x.size());
    d2hdx2.setZero();
    // TODO: Set non-zero elements of d2hdx2
    const double r   = h(0);
    // d2r/dh2 = r1^2 / r^3
    const double d2  = (r1*r1) / (r*r*r);
    d2hdx2(0, 0, 0) = d2;   // only non-zero element

    return h;
}

GaussianInfo<double> MeasurementRADAR::noiseDensity(const SystemEstimator & system) const
{
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    Eigen::MatrixXd SR(1, 1);
    // TODO
    SR(0, 0) = 50.0;  // Standard deviation of measurement noise [m]
    return GaussianInfo<double>::fromSqrtMoment(SR);
}

