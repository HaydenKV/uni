#include <doctest/doctest.h>
#include <cstddef> // for std::size_t
#include <cmath>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

SCENARIO("GaussianInfo simulation")
{
    GIVEN("Almost zero sqrt covariance")
    {
        Eigen::VectorXd mu(2);
        mu << 3, 1;
        Eigen::MatrixXd S(2, 2);
        S << 1e-20, 0,
             0, 1e-20;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        
        WHEN("Evaluating x = p.simulate()")
        {
            Eigen::VectorXd x = p.simulate();
            THEN("x matches expected value")
            {
                CAPTURE_EIGEN(x);
                CAPTURE_EIGEN(mu);
                CHECK(x.isApprox(mu));
            }
        }
    }

    GIVEN("Diagonal sqrt covariance")
    {
        Eigen::VectorXd mu(2);
        mu << 3, 1;
        Eigen::MatrixXd S(2, 2);
        S << 5, 0,
             0, 2;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        
        WHEN("Evaluating 100000 samples from p.simulate()")
        {
            const std::size_t N = 100000;
            Eigen::MatrixXd X(2, N);
            for (std::size_t i = 0; i < N; ++i)
            {
                X.col(i) = p.simulate();
            }

            THEN("Samples have the expected mean")
            {
                Eigen::VectorXd mu_hat = X.rowwise().mean();
                REQUIRE(mu_hat.size() == 2);

                CAPTURE_EIGEN(mu_hat);
                CAPTURE_EIGEN(mu);
                CHECK(mu_hat.isApprox(mu, 0.1));
            }

            THEN("Samples have the expected covariance")
            {
                Eigen::MatrixXd deltaX = X.colwise() - X.rowwise().mean();
                REQUIRE(deltaX.rows() == 2);
                REQUIRE(deltaX.cols() == N);

                Eigen::MatrixXd P_hat = (deltaX*deltaX.transpose())/(N - 1.0);
                REQUIRE(P_hat.rows() == 2);
                REQUIRE(P_hat.cols() == 2);

                Eigen::MatrixXd P = S.transpose()*S;
                CAPTURE_EIGEN(P_hat);
                CAPTURE_EIGEN(P);
                CHECK(P_hat.isApprox(P, 0.1));
            }
        }
    }

    GIVEN("Upper triangular sqrt covariance")
    {
        Eigen::VectorXd mu(2);
        mu << 3, 1;
        Eigen::MatrixXd S(2, 2);
        S << 1, -0.3,
             0,  0.1;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        
        WHEN("Evaluating 100000 samples from p.simulate()")
        {
            const std::size_t N = 100000;
            Eigen::MatrixXd X(2, N);
            for (std::size_t i = 0; i < N; ++i)
            {
                X.col(i) = p.simulate();
            }

            THEN("Samples have the expected mean")
            {
                Eigen::VectorXd mu_hat = X.rowwise().mean();
                REQUIRE(mu_hat.size() == 2);

                CAPTURE_EIGEN(mu_hat);
                CAPTURE_EIGEN(mu);
                CHECK(mu_hat.isApprox(mu, 0.1));
            }

            THEN("Samples have the expected covariance")
            {
                Eigen::MatrixXd deltaX = X.colwise() - X.rowwise().mean();
                REQUIRE(deltaX.rows() == 2);
                REQUIRE(deltaX.cols() == N);

                Eigen::MatrixXd P_hat = (deltaX*deltaX.transpose())/(N - 1.0);
                REQUIRE(P_hat.rows() == 2);
                REQUIRE(P_hat.cols() == 2);

                Eigen::MatrixXd P = S.transpose()*S;
                CAPTURE_EIGEN(P_hat);
                CAPTURE_EIGEN(P);
                CHECK(P_hat.isApprox(P, 0.1));
            }
        }
    }
}
