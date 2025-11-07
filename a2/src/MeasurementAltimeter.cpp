#include "MeasurementAltimeter.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"
#include <Eigen/Core>
#include <cmath>

MeasurementAltimeter::MeasurementAltimeter(double time, double altitude, double sigmaAlt, const Camera & camera)
    : Measurement(time)
    , altitude_(altitude)
    , sigma_(sigmaAlt)
    , camera_(camera)
{
}

MeasurementAltimeter * MeasurementAltimeter::clone() const
{
    return new MeasurementAltimeter(*this);
}

MeasurementAltimeter::~MeasurementAltimeter() = default;

Eigen::VectorXd MeasurementAltimeter::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    // Simulate the altimeter measurement: y = -e3^T rCNn + v  where rCNn is camera position in nav frame
    Eigen::VectorXd y(1);
    Eigen::MatrixXd J;
    // system may be either SystemVisualNav or other estimator; use cameraPosition helper
    Eigen::VectorXd xcopy = x; // cameraPosition expects VectorXd
    Eigen::MatrixXd Jcam;
    Eigen::Vector3d rCNn = SystemVisualNav::cameraPosition(camera_, xcopy, Jcam);
    y(0) = -rCNn(2);
    return y;
}

static double altLogLikelihoodScalar(double ymeas, double ypred, double sigma)
{
    const double r = ymeas - ypred;
    return -0.5 * (r * r) / (sigma * sigma);
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    // Use SystemVisualNav cameraPosition to compute predicted altitude
    Eigen::MatrixXd Jcam;
    Eigen::Vector3d rCNn = SystemVisualNav::cameraPosition(camera_, x, Jcam);
    double ypred = -rCNn(2);
    return altLogLikelihoodScalar(altitude_, ypred, sigma_);
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Compute scalar likelihood and gradient
    const Eigen::Index nx = x.size();
    g.setZero(nx);

    Eigen::MatrixXd Jcam;
    Eigen::Vector3d rCNn = SystemVisualNav::cameraPosition(camera_, x, Jcam);
    double ypred = -rCNn(2);
    double r = altitude_ - ypred;
    double invVar = 1.0 / (sigma_ * sigma_);

    // ∂ logL / ∂x = invVar * r * ∂ypred/∂x ; ypred = -e3^T rCNn so ∂ypred/∂x = - [0 0 1] * Jcam
    Eigen::RowVectorXd dy_dx = -Eigen::RowVectorXd(Jcam.row(2));
    g = invVar * r * dy_dx.transpose();

    return altLogLikelihoodScalar(altitude_, ypred, sigma_);
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    const Eigen::Index nx = x.size();
    g.setZero(nx);
    H.setZero(nx, nx);

    Eigen::MatrixXd Jcam;
    Eigen::Vector3d rCNn = SystemVisualNav::cameraPosition(camera_, x, Jcam);
    double ypred = -rCNn(2);
    double r = altitude_ - ypred;
    double invVar = 1.0 / (sigma_ * sigma_);

    Eigen::RowVectorXd dy_dx = -Eigen::RowVectorXd(Jcam.row(2));
    g = invVar * r * dy_dx.transpose();

    // Hessian (Gauss-Newton approx): H ≈ -invVar * dy_dx^T * dy_dx
    H.noalias() = -invVar * dy_dx.transpose() * dy_dx;

    return altLogLikelihoodScalar(altitude_, ypred, sigma_);
}

void MeasurementAltimeter::update(SystemBase & system)
{
    // For altimeter updates we want to perform a full measurement update (calls base class implementation)
    Measurement::update(system);
}
