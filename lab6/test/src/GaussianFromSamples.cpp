#include <doctest/doctest.h>
#include <cstddef>
#include <Eigen/Core>
#include "../../src/Gaussian.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("Gaussian from samples")
{
    GIVEN("A set of sample points")
    {
        const int n = 2;
        Eigen::MatrixXd X(n, 5);
        X << -0.15508, -1.0443, -1.1714, 0.92622, -0.55806,
             0.61212, -0.34563, -0.68559, -1.4817, -0.028453;

        WHEN("Creating a Gaussian from these samples")
        {
            auto p = Gaussian<double>::fromSamples(X);

            THEN("mu has correct dimensions")
            {
                Eigen::VectorXd mu = p.mean();
                REQUIRE(mu.size() == n);

                AND_THEN("mu matches expected result")
                {
                    Eigen::VectorXd mu_expected(n);
                    mu_expected << -0.40052, -0.38585;
                    
                    CAPTURE_EIGEN(mu);
                    CAPTURE_EIGEN(mu_expected);
                    CHECK(mu.isApprox(mu_expected, 1e-5));
                }
            }

            THEN("S has correct dimensions")
            {
                Eigen::MatrixXd S = p.sqrtCov();
                REQUIRE(S.cols() == n);

                AND_THEN("S is upper triangular")
                {
                    CAPTURE_EIGEN(S);
                    CHECK(S.isUpperTriangular());
                }

                AND_THEN("P matches expected result")
                {
                    Eigen::MatrixXd P = p.cov();
                    Eigen::MatrixXd P_expected(n, n);
                    P_expected <<   0.7135, -0.26502,
                                  -0.26502,  0.60401;
                    
                    CAPTURE_EIGEN(P);
                    CAPTURE_EIGEN(P_expected);
                    CHECK(P.isApprox(P_expected, 1e-5));
                }
            }
        }
    }
}
