#ifndef MEASUREMENTALTIMETER_H
#define MEASUREMENTALTIMETER_H

#include "Measurement.h"
#include "Camera.h"
#include "GaussianInfo.hpp"

class MeasurementAltimeter : public Measurement
{
public:
    MeasurementAltimeter(double time, double altitude, double sigmaAlt, const Camera & camera);
    virtual MeasurementAltimeter * clone() const;
    virtual ~MeasurementAltimeter() override;

    // Implement Measurement interface
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

protected:
    virtual void update(SystemBase & system) override; // use base class implementation which calls the optimizer

private:
    double altitude_;   // measured altitude (m)
    double sigma_;      // measurement noise std (m)
    const Camera & camera_;
};

#endif
