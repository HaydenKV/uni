#include <cmath>
#include <cassert>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemBallistic.h"

const double SystemBallistic::p0 = 101.325e3;            // Air pressure at sea level [Pa]
const double SystemBallistic::M  = 0.0289644;            // Molar mass of dry air [kg/mol]
const double SystemBallistic::R  = 8.31447;              // Gas constant [J/(mol.K)]
const double SystemBallistic::L  = 0.0065;               // Temperature gradient [K/m]
const double SystemBallistic::T0 = 288.15;               // Temperature at sea level [K]
const double SystemBallistic::g  = 9.81;                 // Acceleration due to gravity [m/s^2]

SystemBallistic::SystemBallistic(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{}

GaussianInfo<double> SystemBallistic::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    Eigen::MatrixXd SQ(2, 2);
    // TODO
    SQ.setZero();
    SQ(0,0) = 1.0e-10; // sqrt(1e-20)  -> velocity equation
    SQ(1,1) = 5.0e-6;  // sqrt(25e-12) -> beta equation

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemBallistic::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ;
    // TODO: Continuous-time process noise in 2nd and 3rd state equations
    // no noise in h (0 based indexing)
    idxQ.reserve(2);
    idxQ.push_back(1);  // x2 dynamics
    idxQ.push_back(2);  // x3 dynamics

    return idxQ;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemBallistic::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    assert(x.size() == 3);
    const double h = x(0);
    const double v = x(1);
    const double beta = x(2);

    // params
    const double T  = T0 - L * h;
    const double expn = g * M / (R * L);
    const double p   = p0 * std::pow(T / T0, expn);
    const double rho = (p * M) / (R * T);

    // Drag (quadratic) term with 0.5*rho and v|v|
    const double drag = 0.5 * rho * beta * v * std::abs(v);

    Eigen::VectorXd f(x.size());
    // TODO: Set f
    f(0) = v; //ḣ = v
    f(1) = -g - drag; //v̇ = d - g, and d enters negative because drag opposes motion (sign via v|v|) // if 𝑣<0 (descending), then 𝑣∣𝑣∣=−𝑣2,so drag always opposes motion.
    f(2) = 0.0;
    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemBallistic::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(t, x, u);

    assert(f.size() == 3);
    assert(x.size() == 3);

    const double h = x(0);
    const double v = x(1);
    const double beta = x(2);

    // params
    const double T  = T0 - L * h;
    const double expn = g * M / (R * L);
    const double p   = p0 * std::pow(T / T0, expn);
    const double rho = (p * M) / (R * T);

    // d rho / d h
    const double drho_dh = rho * (expn - 1.0) * (-L) / T;

    // Helpful factor
    const double q = 0.5 * rho;

    J.resize(f.size(), x.size());
    J.setZero();
    // TODO: Set J
    // df1/dx = [0, 1, 0]
    J(0,0) = 0.0;
    J(0,1) = 1.0;
    J(0,2) = 0.0;

    // f2 = -g - 0.5*rho*beta*v|v|
    // ∂f2/∂h = -0.5*(∂rho/∂h)*beta*v|v|
    J(1,0) = -0.5 * drho_dh * beta * v * std::abs(v);

    // ∂f2/∂v = -0.5*rho*beta*d/dv[v|v|] = -0.5*rho*beta*(2|v|) = -rho*beta*|v|
    J(1,1) = -rho * beta * std::abs(v);

    // ∂f2/∂beta = -0.5*rho * v|v|
    J(1,2) = -q * v * std::abs(v);

    // f3 = 0 → row is zeros
    return f;
}

Eigen::VectorXd SystemBallistic::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

