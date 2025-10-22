#include <Eigen/Core>
#include "vnorm.h"

// Templated implementation of vector norm
template<typename Scalar>
static Scalar vnorm(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & x)
{   
    using std::sqrt;
    return sqrt(x.cwiseProduct(x).sum());
}

// Functor for vector norm and its derivatives
double VectorNorm::operator()(const Eigen::VectorXd & x)
{
    return vnorm(x);
}

double VectorNorm::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{
    double u = operator()(x);
    Eigen::Index n = x.size();
    g.resize(n, 1);
    g = x/u;
    return u;
}

double VectorNorm::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    double u = operator()(x, g);
    Eigen::Index n = x.size();
    H.resize(n, n);
    H = Eigen::MatrixXd::Identity(n, n)/u - 1.0/(u*u*u)*x*x.transpose();
    return u;
}

// Functor for vector norm and its derivatives using forward-mode autodifferentiation
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

double VectorNormFwdAutoDiff::operator()(const Eigen::VectorXd & x)
{   
    return vnorm(x);
}

double VectorNormFwdAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{
    // Forward-mode autodifferentiation
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(vnorm<autodiff::dual>, wrt(xdual), at(xdual), fdual);
    return val(fdual);
}

double VectorNormFwdAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    // Forward-mode autodifferentiation
    Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = hessian(vnorm<autodiff::dual2nd>, wrt(xdual), at(xdual), fdual, g);
    return val(fdual);
}

// Functor for vector norm and its derivatives using reverse-mode autodifferentiation
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

double VectorNormRevAutoDiff::operator()(const Eigen::VectorXd & x)
{   
    return vnorm(x);
}

double VectorNormRevAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{
    // Reverse-mode autodifferentiation
    Eigen::VectorX<autodiff::var> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = vnorm(xvar);
    g = gradient(fvar, xvar);
    return val(fvar);
}

double VectorNormRevAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    // Reverse-mode autodifferentiation
    Eigen::VectorX<autodiff::var> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = vnorm(xvar);
    H = hessian(fvar, xvar, g);
    return val(fvar);
}