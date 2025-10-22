#ifndef ROSENBROCK_H
#define ROSENBROCK_H

#include <Eigen/Core>

// Functors for Rosenbrock function and its derivatives

// Functor for Rosenbrock function and its analytical derivatives
struct RosenbrockAnalytical
{
    double operator()(const Eigen::VectorXd & x);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H);
};

// Functor for Rosenbrock function and its derivatives using forward-mode autodifferentiation
struct RosenbrockFwdAutoDiff
{
    double operator()(const Eigen::VectorXd & x);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H);
};

// Functor for Rosenbrock function and its derivatives using reverse-mode autodifferentiation
struct RosenbrockRevAutoDiff
{
    double operator()(const Eigen::VectorXd & x);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H);
};

#endif
