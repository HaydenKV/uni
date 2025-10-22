#include <doctest/doctest.h>
#include <Eigen/Core>
#include "../../src/Gaussian.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("Gaussian addition")
{
    GIVEN("Two Gaussians with S1 = I and S2 = 0")
    {
        Eigen::MatrixXd S1, S2;
        S1 = Eigen::MatrixXd::Identity(3,3);
        S2 = Eigen::MatrixXd::Zero(3,3);
        REQUIRE(S1.cols() == S2.cols());
        Eigen::VectorXd mu1, mu2;
        mu1 = Eigen::VectorXd::Constant(3, 1.0);
        mu2 = Eigen::VectorXd::Constant(3, 2.0);
        REQUIRE(mu1.size() == mu2.size());
        Gaussian p1(mu1, S1);
        Gaussian p2(mu2, S2);

        WHEN("The Gaussian random variables are added")
        {
            Gaussian p = p1.add(p2);

            THEN("mu has expected dimensions")
            {
                Eigen::VectorXd mu = p.mean();
                REQUIRE(mu.size() == mu1.size());

                AND_THEN("mu = mu1 + mu2")
                {
                    Eigen::VectorXd mu_expected = mu1 + mu2;
                    CAPTURE_EIGEN(mu);
                    CAPTURE_EIGEN(mu_expected);
                    CHECK(mu.isApprox(mu_expected));
                }
            }

            THEN("S has expected dimensions")
            {
                Eigen::MatrixXd S = p.sqrtCov();
                //REQUIRE(S.rows() == S.cols());
                REQUIRE(S.cols() == mu1.size());

                AND_THEN("S is upper triangular")
                {
                    CAPTURE_EIGEN(S);
                    CHECK(S.isUpperTriangular());
                }

                AND_THEN("P = P1 + P2")
                {
                    Eigen::MatrixXd P = p.cov();
                    Eigen::MatrixXd P_expected = p1.cov() + p2.cov();
                    CAPTURE_EIGEN(P);
                    CAPTURE_EIGEN(P_expected);
                    CHECK(P.isApprox(P_expected));
                }
            }
        }
    }

    GIVEN("Two Gaussians with S1 = 0 and S2 = I")
    {
        Eigen::MatrixXd S1, S2;
        S1 = Eigen::MatrixXd::Zero(3,3);
        S2 = Eigen::MatrixXd::Identity(3,3);
        REQUIRE(S1.cols() == S2.cols());
        Eigen::VectorXd mu1, mu2;
        mu1 = Eigen::VectorXd::Constant(3, 1.0);
        mu2 = Eigen::VectorXd::Constant(3, 2.0);
        REQUIRE(mu1.size() == mu2.size());
        Gaussian p1(mu1, S1);
        Gaussian p2(mu2, S2);

        WHEN("The Gaussian random variables are added")
        {
            Gaussian p = p1.add(p2);

            THEN("mu has expected dimensions")
            {
                Eigen::VectorXd mu = p.mean();
                REQUIRE(mu.size() == mu1.size());

                AND_THEN("mu = mu1 + mu2")
                {
                    Eigen::VectorXd mu_expected = mu1 + mu2;
                    CAPTURE_EIGEN(mu);
                    CAPTURE_EIGEN(mu_expected);
                    CHECK(mu.isApprox(mu_expected));
                }
            }

            THEN("S has expected dimensions")
            {
                Eigen::MatrixXd S = p.sqrtCov();
                //REQUIRE(S.rows() == S.cols());
                REQUIRE(S.cols() == mu1.size());

                AND_THEN("S is upper triangular")
                {
                    CAPTURE_EIGEN(S);
                    CHECK(S.isUpperTriangular());
                }

                AND_THEN("P = P1 + P2")
                {
                    Eigen::MatrixXd P = p.cov();
                    Eigen::MatrixXd P_expected = p1.cov() + p2.cov();
                    CAPTURE_EIGEN(P);
                    CAPTURE_EIGEN(P_expected);
                    CHECK(P.isApprox(P_expected));
                }
            }
        }
    }

    GIVEN("Two Gaussians with S1 = 3I and S2 = 4I")
    {
        Eigen::MatrixXd S1, S2;
        S1 = 3*Eigen::MatrixXd::Identity(3,3);
        S2 = 4*Eigen::MatrixXd::Identity(3,3);
        REQUIRE(S1.cols() == S2.cols());
        Eigen::VectorXd mu1, mu2;
        mu1 = Eigen::VectorXd::Constant(3, 1.0);
        mu2 = Eigen::VectorXd::Constant(3, 2.0);
        REQUIRE(mu1.size() == mu2.size());
        Gaussian p1(mu1, S1);
        Gaussian p2(mu2, S2);

        WHEN("The Gaussian random variables are added")
        {
            Gaussian p = p1.add(p2);

            THEN("mu has expected dimensions")
            {
                Eigen::VectorXd mu = p.mean();
                REQUIRE(mu.size() == mu1.size());

                AND_THEN("mu = mu1 + mu2")
                {
                    Eigen::VectorXd mu_expected = mu1 + mu2;
                    CAPTURE_EIGEN(mu);
                    CAPTURE_EIGEN(mu_expected);
                    CHECK(mu.isApprox(mu_expected));
                }
            }

            THEN("S has expected dimensions")
            {
                Eigen::MatrixXd S = p.sqrtCov();
                //REQUIRE(S.rows() == S.cols());
                REQUIRE(S.cols() == mu1.size());

                AND_THEN("S is upper triangular")
                {
                    CAPTURE_EIGEN(S);
                    CHECK(S.isUpperTriangular());
                }

                AND_THEN("P = P1 + P2")
                {
                    Eigen::MatrixXd P = p.cov();
                    Eigen::MatrixXd P_expected = p1.cov() + p2.cov();
                    CAPTURE_EIGEN(P);
                    CAPTURE_EIGEN(P_expected);
                    CHECK(P.isApprox(P_expected));
                }
            }
        }
    }

    GIVEN("Two Gaussians with different numbers of rows in S1 and S2")
    {
        Eigen::MatrixXd S1(4,4), S2(1,4);
        S1 << 1, 2, 3, 4,
              0, 5, 6, 7,
              0, 0, 8, 9,
              0, 0, 0,16;
        S2 << 0, 10, -1, -3;
        REQUIRE(S1.cols() == S2.cols());
        Eigen::VectorXd mu1, mu2;
        mu1 = Eigen::VectorXd::Constant(4, 1.0);
        mu2 = Eigen::VectorXd::Constant(4, 2.0);
        REQUIRE(mu1.size() == mu2.size());
        Gaussian p1(mu1, S1);
        Gaussian p2(mu2, S2);

        WHEN("The Gaussian random variables are added")
        {
            Gaussian p = p1.add(p2);

            THEN("mu has expected dimensions")
            {
                Eigen::VectorXd mu = p.mean();
                REQUIRE(mu.size() == mu1.size());

                AND_THEN("mu = mu1 + mu2")
                {
                    Eigen::VectorXd mu_expected = mu1 + mu2;
                    CAPTURE_EIGEN(mu);
                    CAPTURE_EIGEN(mu_expected);
                    CHECK(mu.isApprox(mu_expected));
                }
            }
            
            THEN("S has expected dimensions")
            {
                Eigen::MatrixXd S = p.sqrtCov();
                //REQUIRE(S.rows() == S.cols());
                REQUIRE(S.cols() == mu1.size());

                AND_THEN("S is upper triangular")
                {
                    CAPTURE_EIGEN(S);
                    CHECK(S.isUpperTriangular());
                }

                AND_THEN("P = P1 + P2")
                {
                    Eigen::MatrixXd P = p.cov();
                    Eigen::MatrixXd P_expected = p1.cov() + p2.cov();
                    CAPTURE_EIGEN(P);
                    CAPTURE_EIGEN(P_expected);
                    CHECK(P.isApprox(P_expected));
                }
            }
        }
    }
}
