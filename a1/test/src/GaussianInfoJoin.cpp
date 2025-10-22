#include <doctest/doctest.h>
#include <cstddef>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "../../src/GaussianInfo.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("GaussianInfo join operations")
{
    GIVEN("Two GaussianInfo objects")
    {
        const std::size_t n1 = 3;
        const std::size_t n2 = 2;
        
        Eigen::VectorXd mu1(n1);
        mu1 << 1, 2, 3;
        Eigen::MatrixXd S1(n1, n1);
        S1 << 2, 1, 0,
              0, 3, 1,
              0, 0, 4;
        
        Eigen::VectorXd mu2(n2);
        mu2 << 4, 5;
        Eigen::MatrixXd S2(n2, n2);
        S2 << 1, 0,
              0, 2;

        GaussianInfo<double> p1 = GaussianInfo<double>::fromSqrtMoment(mu1, S1);
        GaussianInfo<double> p2 = GaussianInfo<double>::fromSqrtMoment(mu2, S2);

        WHEN("Joining the two GaussianInfo objects")
        {
            GaussianInfo<double> joined = p1.join(p2);

            THEN("The resulting GaussianInfo has the correct dimension")
            {
                REQUIRE(joined.dim() == n1 + n2);

                AND_THEN("The mean of the joined distribution is correct")
                {
                    Eigen::VectorXd expected_mean(n1 + n2);
                    expected_mean << mu1, mu2;
                    Eigen::VectorXd actual_mean = joined.mean();
                    CAPTURE_EIGEN(actual_mean);
                    CAPTURE_EIGEN(expected_mean);
                    CHECK(actual_mean.isApprox(expected_mean));
                }

                AND_THEN("The covariance of the joined distribution is block diagonal")
                {
                    Eigen::MatrixXd expected_cov = Eigen::MatrixXd::Zero(n1 + n2, n1 + n2);
                    expected_cov.topLeftCorner(n1, n1) = p1.cov();
                    expected_cov.bottomRightCorner(n2, n2) = p2.cov();
                    Eigen::MatrixXd actual_cov = joined.cov();
                    CAPTURE_EIGEN(actual_cov);
                    CAPTURE_EIGEN(expected_cov);
                    CHECK(actual_cov.isApprox(expected_cov));
                }
            }
        }

        WHEN("Multiplying the two GaussianInfo objects using operator*")
        {
            GaussianInfo<double> joined = p1*p2;

            THEN("The resulting GaussianInfo has the correct dimension")
            {
                REQUIRE(joined.dim() == n1 + n2);

                AND_THEN("The mean of the joined distribution is correct")
                {
                    Eigen::VectorXd expected_mean(n1 + n2);
                    expected_mean << mu1, mu2;
                    Eigen::VectorXd actual_mean = joined.mean();
                    CAPTURE_EIGEN(actual_mean);
                    CAPTURE_EIGEN(expected_mean);
                    CHECK(actual_mean.isApprox(expected_mean));
                }

                AND_THEN("The covariance of the joined distribution is block diagonal")
                {
                    Eigen::MatrixXd expected_cov = Eigen::MatrixXd::Zero(n1 + n2, n1 + n2);
                    expected_cov.topLeftCorner(n1, n1) = p1.cov();
                    expected_cov.bottomRightCorner(n2, n2) = p2.cov();
                    Eigen::MatrixXd actual_cov = joined.cov();
                    CAPTURE_EIGEN(actual_cov);
                    CAPTURE_EIGEN(expected_cov);
                    CHECK(actual_cov.isApprox(expected_cov));
                }
            }
        }

        WHEN("Using operator*= to multiply and assign")
        {
            GaussianInfo<double> joined = p1;
            joined *= p2;

            THEN("The resulting GaussianInfo has the correct dimension")
            {
                REQUIRE(joined.dim() == n1 + n2);

                AND_THEN("The mean of the joined distribution is correct")
                {
                    Eigen::VectorXd expected_mean(n1 + n2);
                    expected_mean << mu1, mu2;
                    Eigen::VectorXd actual_mean = joined.mean();
                    CAPTURE_EIGEN(actual_mean);
                    CAPTURE_EIGEN(expected_mean);
                    CHECK(actual_mean.isApprox(expected_mean));
                }

                AND_THEN("The covariance of the joined distribution is block diagonal")
                {
                    Eigen::MatrixXd expected_cov = Eigen::MatrixXd::Zero(n1 + n2, n1 + n2);
                    expected_cov.topLeftCorner(n1, n1) = p1.cov();
                    expected_cov.bottomRightCorner(n2, n2) = p2.cov();
                    Eigen::MatrixXd actual_cov = joined.cov();
                    CAPTURE_EIGEN(actual_cov);
                    CAPTURE_EIGEN(expected_cov);
                    CHECK(actual_cov.isApprox(expected_cov));
                }
            }
        }
    }
}
