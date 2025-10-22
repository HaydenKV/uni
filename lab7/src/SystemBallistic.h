/**
 * @file SystemBallistic.h
 * @brief Defines the SystemBallistic class for ballistic trajectory estimation.
 */

#ifndef SYSTEMBALLISTIC_H
#define SYSTEMBALLISTIC_H

#include <vector>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"

/**
 * @class SystemBallistic
 * @brief Class for ballistic system state estimation.
 *
 * This class extends SystemEstimator to provide functionality specific to ballistic trajectories.
 */
class SystemBallistic : public SystemEstimator
{
public:
    /**
     * @brief Construct a new SystemBallistic object with initial density.
     * @param density Initial state density.
     */
    explicit SystemBallistic(const GaussianInfo<double> & density);

    /**
     * @brief Compute the system dynamics for the ballistic system.
     * @param t Time.
     * @param x State vector.
     * @param u Input vector.
     * @return Eigen::VectorXd The computed dynamics (state derivative).
     */
    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const override;

    /**
     * @brief Compute the system dynamics and its Jacobian for the ballistic system.
     * @param t Time.
     * @param x State vector.
     * @param u Input vector.
     * @param J Output parameter for the Jacobian matrix (w.r.t. x).
     * @return Eigen::VectorXd The computed dynamics (state derivative).
     */
    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const override;

    virtual Eigen::VectorXd input(double t, const Eigen::VectorXd & x) const override;
protected:
    /**
     * @brief Compute the process noise density for the ballistic system.
     * @param dt The time step.
     * @return The process noise density.
     */
    virtual GaussianInfo<double> processNoiseDensity(double dt) const override;

    /**
     * @brief Get the indices of state variables affected by process noise in the ballistic system.
     * @return std::vector<Eigen::Index> The indices of affected state variables.
     */
    virtual std::vector<Eigen::Index> processNoiseIndex() const override;

    // System constants
    static const double p0;  ///< Sea level atmospheric pressure (Pa)
    static const double M;   ///< Molar mass of air (kg/mol)
    static const double R;   ///< Universal gas constant (J/(mol·K))
    static const double L;   ///< Temperature lapse rate (K/m)
    static const double T0;  ///< Sea level standard temperature (K)
    static const double g;   ///< Gravitational acceleration (m/s^2)
};

#endif
