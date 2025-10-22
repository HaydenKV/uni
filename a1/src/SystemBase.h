/**
 * @file SystemBase.h
 * @brief Defines the base class for system representation.
 */

#ifndef SYSTEMBASE_H
#define SYSTEMBASE_H

#include <Eigen/Core>

/**
 * @class SystemBase
 * @brief Base class for system representation.
 *
 * This class provides a basic interface for system dynamics and prediction.
 */
class SystemBase
{
public:
    /**
     * @brief Construct a new SystemBase object.
     */
    SystemBase();

    /**
     * @brief Destroy the SystemBase object.
     */
    virtual ~SystemBase();

    /**
     * @brief Predict the system state at a given time.
     * @param time The time to predict the system state for.
     */
    virtual void predict(double time) = 0;

    /**
     * @brief Compute the system dynamics.
     * @param t The time to evaluate the dynamics at.
     * @param x The state to evaluate the dynamics at.
     * @param u The input to evaluate the dynamics at.
     * @return The computed dynamics (state derivative).
     */
    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const = 0;

    /**
     * @brief Compute the system dynamics and its Jacobian.
     * @param t The time to evaluate the dynamics at.
     * @param x The state to evaluate the dynamics at.
     * @param u The input to evaluate the dynamics at.
     * @param J Output parameter for the Jacobian matrix.
     * @return The computed dynamics (state derivative).
     */
    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const = 0;

    /**
     * @brief Compute the system input
     * @param t The time to evaluate the input at.
     * @param x The state to evaluate the input at.
     * @return The computed input signal.
     */
    virtual Eigen::VectorXd input(double t, const Eigen::VectorXd & x) const = 0;

protected:
    double time_;  ///< The current system time.
};

#endif
