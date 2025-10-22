/**
 * @file MeasurementRADAR.h
 * @brief Defines the MeasurementRADAR class for RADAR measurements in the system.
 */
#ifndef MEASUREMENTRADAR_H
#define MEASUREMENTRADAR_H

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Gaussian.hpp"
#include "SystemEstimator.h"
#include "MeasurementGaussianLikelihood.h"

/**
 * @class MeasurementRADAR
 * @brief Class for RADAR measurements.
 *
 * This class represents RADAR measurements in the system.
 * It inherits from MeasurementGaussianLikelihood and implements RADAR-specific functionality.
 */
class MeasurementRADAR : public MeasurementGaussianLikelihood
{
public:
    /**
     * @brief Construct a new MeasurementRADAR object.
     * @param time The time at which the measurement occurs.
     * @param y The measurement vector.
     */
    MeasurementRADAR(double time, const Eigen::VectorXd & y);

    /**
     * @brief Construct a new MeasurementRADAR object.
     * @param time The time at which the measurement occurs.
     * @param y The measurement vector.
     * @param verbosity The verbosity level for logging and output.
     */
    MeasurementRADAR(double time, const Eigen::VectorXd & y, int verbosity);

    /**
     * @brief Destroy the MeasurementRADAR object.
     */
    virtual ~MeasurementRADAR() override;

    // Inherited virtual functions
    virtual Eigen::VectorXd predict(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual Eigen::VectorXd predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & J) const override;
    virtual Eigen::VectorXd predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & J, Eigen::Tensor<double, 3> & H) const override;
    virtual Gaussian<double> noiseDensity(const SystemEstimator & system) const override;
protected:
    /**
     * @brief Get a string representation of the RADAR measurement process.
     * @return std::string A string describing the RADAR measurement process.
     */
    virtual std::string getProcessString() const override;

    static const double r1;  ///< RADAR horizontal offset
    static const double r2;  ///< RADAR vertical offset
};

#endif
