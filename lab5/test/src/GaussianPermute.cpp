#include <doctest/doctest.h>
#include <Eigen/Core>
#include "../../src/Gaussian.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("Gaussian permutation")
{
    GIVEN("A Gaussian distribution with a permutation (n = 4)")
    {
        const int n = 4;
        Eigen::VectorXd mu(n);
        mu << 1, 2, 3, 4;

        Eigen::MatrixXd S(n, n);
        S << 1, 2, 3, 4,
             0, 5, 6, 7,
             0, 0, 8, 9,
             0, 0, 0, 10;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        Gaussian p(mu, S);
        Eigen::MatrixXd P = p.cov();
        
        Eigen::ArrayXi idx(n);
        idx << 3, 1, 2, 0;

        WHEN("A permutation is applied")
        {
            Gaussian pI = p.permute(idx);

            THEN("The permuted mean has correct dimensions")
            {
                Eigen::VectorXd muI = pI.mean();
                REQUIRE(muI.size() == n);
            
                AND_THEN("The mean is correctly permuted")
                {
                    Eigen::VectorXd muI_expected = mu(idx);
                    CAPTURE_EIGEN(muI);
                    CAPTURE_EIGEN(muI_expected);
                    CHECK(muI.isApprox(muI_expected));
                }
            }

            THEN("The permuted square root covariance has correct dimensions")
            {
                Eigen::MatrixXd SI = pI.sqrtCov();
                // REQUIRE(SI.rows() == n);
                REQUIRE(SI.cols() == n);

                AND_THEN("The permuted square root covariance is upper triangular")
                {
                    CAPTURE_EIGEN(SI);
                    CHECK(SI.isUpperTriangular());
                }

                AND_THEN("The covariance is correctly permuted")
                {
                    Eigen::MatrixXd PI = pI.cov();
                    Eigen::MatrixXd PI_expected = P(idx, idx);
                    CAPTURE_EIGEN(PI);
                    CAPTURE_EIGEN(PI_expected);
                    CHECK(PI.isApprox(PI_expected));
                }
            }
        }
    }

    GIVEN("A Gaussian distribution with a permutation (n = 2)")
    {
        const int n = 2;
        Eigen::VectorXd mu(n);
        mu << 1, 2;

        Eigen::MatrixXd S(n, n);
        S << 1, 2,
             0, 3;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        Gaussian p(mu, S);
        Eigen::MatrixXd P = p.cov();
        
        Eigen::ArrayXi idx(n);
        idx << 1, 0;

        WHEN("A permutation is applied")
        {
            Gaussian pI = p.permute(idx);

            THEN("The permuted mean has correct dimensions")
            {
                Eigen::VectorXd muI = pI.mean();
                REQUIRE(muI.size() == n);

                AND_THEN("The mean is correctly permuted")
                {
                    Eigen::VectorXd muI_expected = mu(idx);
                    CAPTURE_EIGEN(muI);
                    CAPTURE_EIGEN(muI_expected);
                    CHECK(muI.isApprox(muI_expected));
                }
            }

            THEN("The permuted square root covariance has correct dimensions")
            {
                Eigen::MatrixXd SI = pI.sqrtCov();
                // REQUIRE(SI.rows() == n);
                REQUIRE(SI.cols() == n);

                AND_THEN("The permuted square root covariance is upper triangular")
                {
                    CAPTURE_EIGEN(SI);
                    CHECK(SI.isUpperTriangular());
                }

                AND_THEN("The covariance is correctly permuted")
                {
                    Eigen::MatrixXd PI = pI.cov();
                    Eigen::MatrixXd PI_expected = P(idx, idx);
                    CAPTURE_EIGEN(PI);
                    CAPTURE_EIGEN(PI_expected);
                    CHECK(PI.isApprox(PI_expected));
                }
            }
        }
    }
}

