/**
 * @mainpage MCHA4400 Lab 6: Laplace filter
 *
 * @tableofcontents
 *
 * @section intro Introduction
 *
 * In this lab, you will:
 * - Implement Gaussian distribution operations in square-root moment form
 * - Implement the process dynamics and measurement model for a ballistic state estimation problem
 * - Run a square-root Laplace filter
 * - Plot the results using VTK
 *
 * @section tasks Tasks
 *
 * 1. Implement Gaussian distribution operations (log likelihood, affine transform, marginal, conditional)
 * 2. Implement ballistic process model dynamics 
 * 3. Implement RADAR range measurement model
 * 4. Run Laplace filter and visualise results
 *
 * @section implementation Key Implementation Files
 * 
 * - `src/GaussianBase.hpp`: Base class for Gaussian distribution
 * - `src/Gaussian.hpp`: Gaussian distribution in square-root moment form
 * - `src/SystemBallistic.cpp`: System dynamics for ballistic trajectory
 * - `src/MeasurementRADAR.cpp`: Measurement model and likelihood for RADAR
 * - `src/ballistic_plot.cpp`: Plotting functions for ballistic trajectory
 *
 * @section testing Unit Tests
 *
 * Unit tests are provided to verify your implementations:
 * 
 * - `test/src/GaussianLog.cpp`
 * - `test/src/GaussianLogIntegral.cpp`
 * - `test/src/GaussianTransform.cpp`
 * - `test/src/GaussianMarginal.cpp`
 * - `test/src/GaussianConditional.cpp`
 * - `test/src/GaussianConfidence.cpp`
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
#include <Eigen/Core>
#include "to_string.hpp"
#include "Gaussian.hpp"
#include "SystemBallistic.h"
#include "MeasurementRADAR.h"
#include "ballistic_plot.h"

int main(int argc, char *argv[])
{
    // ------------------------------------------------------
    // Laplace filter
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

    Gaussian<double> p0 = Gaussian<double>::fromSqrtMoment(mu0, S0);
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
