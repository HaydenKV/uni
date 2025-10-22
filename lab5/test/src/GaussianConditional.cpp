#include <doctest/doctest.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "../../src/Gaussian.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("Gaussian conditional")
{
    GIVEN("A joint Gaussian and xa (na = 1, nb = 1)")
    {
        const int na = 1;
        const int nb = 1;

        Eigen::VectorXd mu(na + nb);
        mu << 1,
              1;
        Eigen::VectorXd mua = mu.head(na);
        Eigen::VectorXd mub = mu.tail(nb);

        Eigen::MatrixXd S(na + nb, na + nb);
        S << 1, -0.649013765191241,
             0, 1;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");
        
        Gaussian p(mu, S);
        Eigen::MatrixXd P = p.cov();
        Eigen::MatrixXd Paa = P.topLeftCorner(na, na);
        Eigen::MatrixXd Pab = P.topRightCorner(na, nb);
        Eigen::MatrixXd Pba = P.bottomLeftCorner(nb, na);
        Eigen::MatrixXd Pbb = P.bottomRightCorner(nb, nb);

        Eigen::VectorXd xa(na);
        xa << 0;

        WHEN("p.conditionalTailGivenHead(xa) is called")
        {
            Gaussian pb_a = p.conditionalTailGivenHead(xa);

            THEN("mub_a has correct dimensions")
            {
                Eigen::VectorXd mub_a = pb_a.mean();
                REQUIRE(mub_a.size() == nb);

                AND_THEN("mub_a matches expected result")
                {
                    Eigen::VectorXd mub_a_expected =
                        mub + Pba*Paa.llt().solve(xa - mua);
                    CAPTURE_EIGEN(mub_a);
                    CAPTURE_EIGEN(mub_a_expected);
                    CHECK(mub_a.isApprox(mub_a_expected));
                }
            }

            THEN("Sb_a has correct dimensions")
            {
                Eigen::MatrixXd Sb_a = pb_a.sqrtCov();
                //REQUIRE(Sb_a.rows() == nb);
                REQUIRE(Sb_a.cols() == nb);

                AND_THEN("Sb_a is upper triangular")
                {
                    CAPTURE_EIGEN(Sb_a);
                    CHECK(Sb_a.isUpperTriangular());
                }

                AND_THEN("Pb_a matches expected result")
                {
                    Eigen::MatrixXd Pb_a = pb_a.cov();
                    Eigen::MatrixXd Pb_a_expected =
                        Pbb - Pba*Paa.llt().solve(Pab);
                    CAPTURE_EIGEN(Pb_a);
                    CAPTURE_EIGEN(Pb_a_expected);
                    CHECK(Pb_a.isApprox(Pb_a_expected));
                }
            }
        }
    }

    GIVEN("A joint Gaussian and xa (na = 3, nb = 1)")
    {
        const int na = 3;
        const int nb = 1;

        Eigen::VectorXd mu(na + nb);
        mu << 1,
              1,
              1,
              1;
        Eigen::VectorXd mua = mu.head(na);
        Eigen::VectorXd mub = mu.tail(nb);

        Eigen::MatrixXd S(na + nb, na + nb);
        S << -0.649013765191241,  -1.10961303850152, -0.558680764473972,  0.586442621667069,
                              0, -0.845551240007797,  0.178380225849766, -0.851886969622469,
                              0,                  0, -0.196861446475943,  0.800320709801823,
                              0,                  0,                  0,  -1.50940472473439;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");
        
        Gaussian p(mu, S);
        Eigen::MatrixXd P = p.cov();
        Eigen::MatrixXd Paa = P.topLeftCorner(na, na);
        Eigen::MatrixXd Pab = P.topRightCorner(na, nb);
        Eigen::MatrixXd Pba = P.bottomLeftCorner(nb, na);
        Eigen::MatrixXd Pbb = P.bottomRightCorner(nb, nb);

        Eigen::VectorXd xa(na);
        xa << 0.875874147834533,
              -0.24278953633334,
              0.166813439453503;

        WHEN("p.conditionalTailGivenHead(xa) is called")
        {
            Gaussian pb_a = p.conditionalTailGivenHead(xa);

            THEN("mub_a has correct dimensions")
            {
                Eigen::VectorXd mub_a = pb_a.mean();
                REQUIRE(mub_a.size() == nb);

                AND_THEN("mub_a matches expected result")
                {
                    Eigen::VectorXd mub_a_expected =
                        mub + Pba*Paa.llt().solve(xa - mua);
                    CAPTURE_EIGEN(mub_a);
                    CAPTURE_EIGEN(mub_a_expected);
                    CHECK(mub_a.isApprox(mub_a_expected));
                }
            }

            THEN("Sb_a has correct dimensions")
            {
                Eigen::MatrixXd Sb_a = pb_a.sqrtCov();
                //REQUIRE(Sb_a.rows() == nb);
                REQUIRE(Sb_a.cols() == nb);

                AND_THEN("Sb_a is upper triangular")
                {
                    CAPTURE_EIGEN(Sb_a);
                    CHECK(Sb_a.isUpperTriangular());
                }

                AND_THEN("Pb_a matches expected result")
                {
                    Eigen::MatrixXd Pb_a = pb_a.cov();
                    Eigen::MatrixXd Pb_a_expected =
                        Pbb - Pba*Paa.llt().solve(Pab);
                    CAPTURE_EIGEN(Pb_a);
                    CAPTURE_EIGEN(Pb_a_expected);
                    CHECK(Pb_a.isApprox(Pb_a_expected));
                }
            }
        }
    }

    GIVEN("A joint Gaussian and xa (na = 1, nb = 3)")
    {
        const int na = 1;
        const int nb = 3;

        Eigen::VectorXd mu(na + nb);
        mu << 1,
              1,
              1,
              1;
        Eigen::VectorXd mua = mu.head(na);
        Eigen::VectorXd mub = mu.tail(nb);

        Eigen::MatrixXd S(na + nb, na + nb);
        S << -0.649013765191241,   1.18116604196553, -0.758453297283692,  -1.10961303850152,
                              0, -0.845551240007797,  0.178380225849766, -0.851886969622469,
                              0,                  0, -0.196861446475943,  0.800320709801823,
                              0,                  0,                  0,  -1.50940472473439;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");
        
        Gaussian p(mu, S);
        Eigen::MatrixXd P = p.cov();
        Eigen::MatrixXd Paa = P.topLeftCorner(na, na);
        Eigen::MatrixXd Pab = P.topRightCorner(na, nb);
        Eigen::MatrixXd Pba = P.bottomLeftCorner(nb, na);
        Eigen::MatrixXd Pbb = P.bottomRightCorner(nb, nb);

        Eigen::VectorXd xa(na);
        xa << 0.875874147834533;

        WHEN("p.conditionalTailGivenHead(xa) is called")
        {
            Gaussian pb_a = p.conditionalTailGivenHead(xa);

            THEN("mub_a has correct dimensions")
            {
                Eigen::VectorXd mub_a = pb_a.mean();
                REQUIRE(mub_a.size() == nb);

                AND_THEN("mub_a matches expected result")
                {
                    Eigen::VectorXd mub_a_expected =
                        mub + Pba*Paa.llt().solve(xa - mua);
                    CAPTURE_EIGEN(mub_a);
                    CAPTURE_EIGEN(mub_a_expected);
                    CHECK(mub_a.isApprox(mub_a_expected));
                }
            }

            THEN("Sb_a has correct dimensions")
            {
                Eigen::MatrixXd Sb_a = pb_a.sqrtCov();
                //REQUIRE(Sb_a.rows() == nb);
                REQUIRE(Sb_a.cols() == nb);

                AND_THEN("Sb_a is upper triangular")
                {
                    CAPTURE_EIGEN(Sb_a);
                    CHECK(Sb_a.isUpperTriangular());
                }

                AND_THEN("Pb_a matches expected result")
                {
                    Eigen::MatrixXd Pb_a = pb_a.cov();
                    Eigen::MatrixXd Pb_a_expected =
                        Pbb - Pba*Paa.llt().solve(Pab);
                    CAPTURE_EIGEN(Pb_a);
                    CAPTURE_EIGEN(Pb_a_expected);
                    CHECK(Pb_a.isApprox(Pb_a_expected));
                }
            }
        }
    }

    GIVEN("A joint Gaussian and xa (na = 3, nb = 3)")
    {
        const int na = 3;
        const int nb = 3;

        Eigen::VectorXd mu(na + nb);
        mu << 1,
              1,
              1,
              1,
              1,
              1;
        Eigen::VectorXd mua = mu.head(na);
        Eigen::VectorXd mub = mu.tail(nb);

        Eigen::MatrixXd S(na + nb, na + nb);
        S << -0.649013765191241,  -1.10961303850152, -0.558680764473972,  0.586442621667069, -1.50940472473439,  0.166813439453503,
                              0, -0.845551240007797,  0.178380225849766, -0.851886969622469, 0.875874147834533,  -1.96541870928278,
                              0,                  0, -0.196861446475943,  0.800320709801823, -0.24278953633334,  -1.27007139263854,
                              0,                  0,                  0,   1.17517126546302, 0.603658445825815,  -1.86512257453063,
                              0,                  0,                  0,                  0,   1.7812518932425,  -1.05110705924059,
                              0,                  0,                  0,                  0,                 0, -0.417382047996795;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");
        
        Gaussian p(mu, S);
        Eigen::MatrixXd P = p.cov();
        Eigen::MatrixXd Paa = P.topLeftCorner(na, na);
        Eigen::MatrixXd Pab = P.topRightCorner(na, nb);
        Eigen::MatrixXd Pba = P.bottomLeftCorner(nb, na);
        Eigen::MatrixXd Pbb = P.bottomRightCorner(nb, nb);

        Eigen::VectorXd xa(na);
        xa <<   1.40216228633781,
               -1.36774699097611,
              -0.292534999151873;

        WHEN("p.conditionalTailGivenHead(xa) is called")
        {
            Gaussian pb_a = p.conditionalTailGivenHead(xa);

            THEN("mub_a has correct dimensions")
            {
                Eigen::VectorXd mub_a = pb_a.mean();
                REQUIRE(mub_a.size() == nb);

                AND_THEN("mub_a matches expected result")
                {
                    Eigen::VectorXd mub_a_expected =
                        mub + Pba*Paa.llt().solve(xa - mua);
                    CAPTURE_EIGEN(mub_a);
                    CAPTURE_EIGEN(mub_a_expected);
                    CHECK(mub_a.isApprox(mub_a_expected));
                }
            }

            THEN("Sb_a has correct dimensions")
            {
                Eigen::MatrixXd Sb_a = pb_a.sqrtCov();
                //REQUIRE(Sb_a.rows() == nb);
                REQUIRE(Sb_a.cols() == nb);

                AND_THEN("Sb_a is upper triangular")
                {
                    CAPTURE_EIGEN(Sb_a);
                    CHECK(Sb_a.isUpperTriangular());
                }

                AND_THEN("Pb_a matches expected result")
                {
                    Eigen::MatrixXd Pb_a = pb_a.cov();
                    Eigen::MatrixXd Pb_a_expected =
                        Pbb - Pba*Paa.llt().solve(Pab);
                    CAPTURE_EIGEN(Pb_a);
                    CAPTURE_EIGEN(Pb_a_expected);
                    CHECK(Pb_a.isApprox(Pb_a_expected));
                }
            }
        }
    }
}
