#ifndef GAUSSIAN_H
#define GAUSSIAN_H

// Tip: Only include headers needed to parse this header only
#include <Eigen/Core>

class Gaussian
{
public:
    Gaussian();
    Gaussian(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S);
    Eigen::VectorXd mean() const;
    Eigen::MatrixXd sqrtCov() const;
    Eigen::MatrixXd cov() const;
    Gaussian add(const Gaussian & other) const;
    Gaussian marginalHead(int na) const;
    Gaussian conditionalTailGivenHead(const Eigen::VectorXd & xa) const;
    Gaussian permute(const Eigen::ArrayXi &idx) const;
private:
    Eigen::VectorXd mu_;
    Eigen::MatrixXd S_;
};

#endif
