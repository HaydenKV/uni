#include <doctest/doctest.h>
#include <Eigen/Core>
#include "../../src/Gaussian.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("Gaussian marginal")
{
    GIVEN("A joint Gaussian (na = 1, nb = 1)")
    {
        const int na = 1;
        const int nb = 1; (void)nb; // nb is not used in this test

        Eigen::VectorXd mu(2);
        mu << 1,
              2;

        Eigen::MatrixXd S(2, 2);
        S << 1, 0.5,
             0, 1;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        Gaussian p(mu, S);
        Eigen::MatrixXd P = p.cov();

        WHEN("p.marginalHead(na) is called")
        {
            Gaussian pa = p.marginalHead(na);

            THEN("mua has correct dimensions")
            {
                Eigen::VectorXd mua = pa.mean();
                REQUIRE(mua.size() == na);

                AND_THEN("mua matches the first part of joint mean mu")
                {
                    Eigen::VectorXd mua_expected = mu.head(na);
                    CAPTURE_EIGEN(mua);
                    CAPTURE_EIGEN(mua_expected);
                    CHECK(mua.isApprox(mua_expected));
                }
            }

            THEN("Sa has correct dimensions")
            {
                Eigen::MatrixXd Sa = pa.sqrtCov();
                // REQUIRE(Sa.rows() == na);
                REQUIRE(Sa.cols() == na);

                AND_THEN("Sa is upper triangular")
                {
                    CAPTURE_EIGEN(Sa);
                    CHECK(Sa.isUpperTriangular());
                }

                AND_THEN("Pa matches the top-left block of joint covariance P")
                {
                    Eigen::MatrixXd Pa = pa.cov();
                    Eigen::MatrixXd Pa_expected = P.topLeftCorner(na, na);
                    CAPTURE_EIGEN(Pa);
                    CAPTURE_EIGEN(Pa_expected);
                    CHECK(Pa.isApprox(Pa_expected));
                }
            }
        }
    }

    GIVEN("A joint Gaussian (na = 2, nb = 2)")
    {
        const int na = 2;
        const int nb = 2; (void)nb; // nb is not used in this test

        Eigen::VectorXd mu(4);
        mu << 1,
              1,
              1,
              1;

        Eigen::MatrixXd S(4, 4);
        S << 1, 0.5, 0.2, 0.1,
             0, 0.5, 0.2, 0.1,
             0, 0,   0.5, 0.2,
             0, 0,   0,   0.5;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        Gaussian p(mu, S);
        Eigen::MatrixXd P = p.cov();

        WHEN("p.marginalHead(na) is called")
        {
            Gaussian pa = p.marginalHead(na);

            THEN("mua has correct dimensions")
            {
                Eigen::VectorXd mua = pa.mean();
                REQUIRE(mua.size() == na);

                AND_THEN("mua matches the first part of joint mean mu")
                {
                    Eigen::VectorXd mua_expected = mu.head(na);
                    CAPTURE_EIGEN(mua);
                    CAPTURE_EIGEN(mua_expected);
                    CHECK(mua.isApprox(mua_expected));
                }
            }

            THEN("Sa has correct dimensions")
            {
                Eigen::MatrixXd Sa = pa.sqrtCov();
                // REQUIRE(Sa.rows() == na);
                REQUIRE(Sa.cols() == na);

                AND_THEN("Sa is upper triangular")
                {
                    CAPTURE_EIGEN(Sa);
                    CHECK(Sa.isUpperTriangular());
                }

                AND_THEN("Pa matches the top-left block of joint covariance P")
                {
                    Eigen::MatrixXd Pa = pa.cov();
                    Eigen::MatrixXd Pa_expected = P.topLeftCorner(na, na);
                    CAPTURE_EIGEN(Pa);
                    CAPTURE_EIGEN(Pa_expected);
                    CHECK(Pa.isApprox(Pa_expected));
                }
            }
        }
    }
}