/**
 * @mainpage MCHA4400 Lab 7: Laplace information filter
 *
 * @tableofcontents
 *
 * @section intro Introduction
 *
 * In this lab, you will:
 * - Implement optimisation using analytical derivatives and automatic differentiation
 * - Implement Gaussian distribution operations in square-root information form
 * - Implement the process dynamics and measurement model for a ballistic state estimation problem
 * - Run a square-root Laplace information filter
 * - Plot the results using VTK
 *
 * @section tasks Tasks
 *
 * 1. Implement Rosenbrock function and derivatives using analytical and automatic differentiation
 * 2. Implement Gaussian distribution operations (log likelihood, affine transform, marginal, conditional)
 * 3. Implement ballistic process model dynamics 
 * 4. Implement RADAR range measurement model
 * 5. Run Laplace information filter and visualise results
 *
 * @section implementation Key Implementation Files
 * 
 * - `src/rosenbrock.cpp`: Rosenbrock function implementations
 * - `src/GaussianBase.hpp`: Base class for Gaussian distribution
 * - `src/GaussianInfo.hpp`: Gaussian distribution in square-root information form
 * - `src/SystemBallistic.cpp`: System dynamics for ballistic trajectory
 * - `src/MeasurementRADAR.cpp`: Measurement model and likelihood for RADAR
 * - `src/ballistic_plot.cpp`: Plotting functions for ballistic trajectory
 *
 * @section testing Unit Tests
 *
 * Unit tests are provided to verify your implementations:
 * 
 * - `test/src/rosenbrock.cpp`
 * - `test/src/GaussianInfoLog.cpp`
 * - `test/src/GaussianInfoLogIntegral.cpp`
 * - `test/src/GaussianInfoTransform.cpp`
 * - `test/src/GaussianInfoMarginal.cpp`
 * - `test/src/GaussianInfoConditional.cpp`
 * - `test/src/GaussianInfoConfidence.cpp`
 * - `test/src/SystemBallistic.cpp`
 * - `test/src/MeasurementRADAR.cpp`
 *
 * @section build Building and Running
 *
 * 1. Configure CMake build:
 *    `cmake -G Ninja -B build -DBUILD_DOCUMENTATION=ON -DCMAKE_BUILD_TYPE=Debug && cd build`
 * 
 * 2. Build project and run unit tests:
 *    `ninja`
 *
 * 3. Build and run executable:
 *    `ninja && ./lab6`
 */
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <print>
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]") // Format like MATLAB
#include <Eigen/Core>
#include "to_string.hpp"
#include "GaussianInfo.hpp"
#include "SystemBallistic.h"
#include "MeasurementRADAR.h"
#include "ballistic_plot.h"
#include "rosenbrock.h"
#include "funcmin.hpp"

int main(int argc, char *argv[])
{
    // ------------------------------------------------------
    // Optimisation
    // ------------------------------------------------------
    
    Eigen::VectorXd x(2);
    x << 10.0, 10.0;
    std::println("Initial x =\n{}\n", to_string(x));

    // Experiment with the different Rosenbrock functors below
    RosenbrockAnalytical func;
    // RosenbrockFwdAutoDiff func;
    // RosenbrockRevAutoDiff func;

    Eigen::VectorXd g(2);
    Eigen::MatrixXd H(2, 2);
    std::println("f = {}\n", func(x, g, H));
    std::println("g =\n{}\n", to_string(g));
    std::println("H =\n{}\n", to_string(H));

    std::println("Running optimisation");
    int verbosity = 3;  // 0: silent, 1: dots, 2: summary, 3: iteration details
    // funcmin::NewtonTrust(func, x, g, H, verbosity);
    funcmin::BFGSTrust(func, x, g, H, verbosity);
    std::println("\nFinal x =\n{}\n", to_string(x));

    Eigen::MatrixXd H_actual(2, 2);
    std::println("f = {}\n", func(x, g, H_actual));
    std::println("g =\n{}\n", to_string(g));
    std::println("H (actual) =\n{}\n", to_string(H_actual));
    std::println("H (approx) =\n{}\n", to_string(H));

    // Comment out the following line to run the state estimator
    // return EXIT_SUCCESS;

    // ------------------------------------------------------
    // Laplace information filter
    // ------------------------------------------------------

    std::string fileName   = "../data/estimationdata.csv";
    
    // Dimensions of state and measurement vectors for recording results
    const std::size_t nx = 3;
    const std::size_t ny = 1;

    Eigen::VectorXd x0(nx);
    Eigen::VectorXd u;
    Eigen::VectorXd t_hist;
    Eigen::MatrixXd x_hist, y_hist;

    // Read from CSV
    std::fstream input;
    input.open(fileName, std::fstream::in);
    if (!input.is_open())
    {
        std::println("Could not open input file \"{}\"! Exiting", fileName);
        return EXIT_FAILURE;
    }
    std::println("Reading data from {}", fileName);

    // Determine number of time steps
    std::size_t rows = 0;
    std::string line;
    while (std::getline(input, line))
    {
        rows++;
    }
    std::println("Found {} rows within {}\n", rows, fileName);
    std::size_t nsteps = rows - 1;  // Disregard header row

    t_hist.resize(nsteps);
    x_hist.resize(nx, nsteps);
    y_hist.resize(ny, nsteps);

    // Read each row of data
    rows = 0;
    input.clear();
    input.seekg(0);
    std::vector<std::string> row;
    std::string csvElement;
    while (std::getline(input, line))
    {
        if (rows > 0)
        {
            std::size_t i = rows - 1;
            
            row.clear();

            std::stringstream s(line);
            while (std::getline(s, csvElement, ','))
            {
                row.push_back(csvElement);
            }
            
            t_hist(i)    = std::stod(row[0]);
            x_hist(0, i) = std::stod(row[1]);
            x_hist(1, i) = std::stod(row[2]);
            x_hist(2, i) = std::stod(row[3]);
            y_hist(0, i) = std::stod(row[5]);
        }
        rows++;
    }

    Eigen::MatrixXd mu_hist(nx, nsteps);
    Eigen::MatrixXd sigma_hist(nx, nsteps);

    // Initial state estimate
    Eigen::MatrixXd S0(nx, nx);
    Eigen::VectorXd mu0(nx);
    S0.fill(0);
    S0.diagonal() << 2200, 100, 1e-3;

    mu0 << 14000, // Initial height
            -450, // Initial velocity
          0.0005; // Ballistic coefficient

    GaussianInfo<double> p0 = GaussianInfo<double>::fromSqrtMoment(mu0, S0);
    SystemBallistic system(p0);

    std::println("Initial state estimate");
    std::println("mu[0] = \n{}", to_string(p0.mean()));
    std::println("P[0] = \n{}", to_string(p0.cov()));

    for (std::size_t k = 0; k < nsteps; ++k)
    {
        // Create RADAR measurement
        double t = t_hist(k);
        Eigen::VectorXd y = y_hist.col(k);
        MeasurementRADAR measurementRADAR(t, y);

        // Process measurement event (do time update and measurement update)
        measurementRADAR.process(system);

        // Save results for plotting
        mu_hist.col(k)       = system.density.mean();
        sigma_hist.col(k)    = system.density.cov().diagonal().cwiseSqrt();
    }

    std::println("\nFinal state estimate");
    std::println("mu[end] = \n{}", to_string(system.density.mean()));
    std::println("P[end] = \n{}", to_string(system.density.cov()));

    // Plot results
    plot_simulation(t_hist, x_hist, mu_hist, sigma_hist);

    return EXIT_SUCCESS;
}
