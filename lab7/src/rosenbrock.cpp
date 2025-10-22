#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include "rosenbrock.h"

// Templated version of Rosenbrock function
template <typename Scalar = double>
static Scalar rosenbrock(const Eigen::VectorX<Scalar> & x)
{   
    Scalar x2 = x(0)*x(0);
    Scalar ymx2 = x(1) - x2;
    Scalar xm1 = x(0) - 1;
    return (xm1*xm1 + 100*ymx2*ymx2);
}

// Functor for Rosenbrock function and its derivatives
double RosenbrockAnalytical::operator()(const Eigen::VectorXd & x)
{
    return rosenbrock(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{
    // Rosenbrock: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
    // Analytical gradient:
    // df/dx = 2(x - 1) - 400 x (y - x^2)
    // df/dy = 200 (y - x^2)

    // Compute analytical gradient g
    g.resize(2, 1);
    
    // TODO
    const double a = x(1) - x(0)*x(0); // (y - x^2), reused to avoid recomputing
    g(0) = 2.0 * (x(0) - 1.0) - 400.0 * x(0) * a;
    g(1) = 200.0 * a;

    return operator()(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    // Hessian of Rosenbrock:
    // ∂²f/∂x² = 2 - 400(y - x^2) + 800 x^2 = 2 - 400y + 1200 x^2
    // ∂²f/∂x∂y = ∂²f/∂y∂x = -400 x
    // ∂²f/∂y² = 200

    // Compute analytical Hessian H
    H.resize(2, 2);
    // TODO
    H(0,0) = 2.0 - 400.0 * x(1) + 1200.0 * x(0) * x(0);
    H(0,1) = -400.0 * x(0);
    H(1,0) = H(0,1);
    H(1,1) = 200.0;

    return operator()(x, g);
}

// Functor for Rosenbrock function and its derivatives using forward-mode autodifferentiation
double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd & x)
{   
    return rosenbrock(x);
}

double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{
    // Forward-mode autodifferentiation
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();       // lift inputs to dual
    autodiff::dual fdual;                                                  // dual output
    g = gradient(rosenbrock<autodiff::dual>, wrt(xdual), at(xdual), fdual); // compute ∂f/∂x
    return val(fdual);                                                     // scalar value
}

double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    // Forward-mode autodifferentiation
    Eigen::VectorX<autodiff::dual2nd> xdual2 = x.cast<autodiff::dual2nd>(); // lift to dual2nd
    autodiff::dual2nd fdual;                                                // dual2nd output
    H = hessian(rosenbrock<autodiff::dual2nd>, wrt(xdual2), at(xdual2), fdual, g); // Hessian + gradient
    return val(fdual);                                                      // scalar value
}

// Functor for Rosenbrock function and its derivatives using reverse-mode autodifferentiation
double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd & x)
{   
    return rosenbrock(x);
}

double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{
    // Reverse-mode autodifferentiation
    Eigen::VectorX<autodiff::var> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = rosenbrock(xvar);
    // TODO
    g = gradient(fvar, xvar);                                     // backprop for ∂f/∂x
    return val(fvar);                                             // scalar value
}

double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    // Reverse-mode autodifferentiation
    Eigen::VectorX<autodiff::var> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = rosenbrock(xvar);
    // TODO
    H = hessian(fvar, xvar, g);                                   // Hessian + gradient
    return val(fvar); // scalar value
}