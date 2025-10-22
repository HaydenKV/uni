#ifndef VNORM_H
#define VNORM_H

#include <Eigen/Core>

// Functor for vector norm and its derivatives
struct VectorNorm
{
    double operator()(const Eigen::VectorXd & x);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H);
};

// Functor for vector norm and its derivatives using forward-mode autodifferentiation
struct VectorNormFwdAutoDiff
{
    double operator()(const Eigen::VectorXd & x);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H);
};

// Functor for vector norm and its derivatives using reverse-mode autodifferentiation
struct VectorNormRevAutoDiff
{
    double operator()(const Eigen::VectorXd & x);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g);
    double operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H);
};


#endif
