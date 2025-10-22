#include <doctest/doctest.h>
#include <cstddef>
#include <Eigen/Core>
#include "../../src/Gaussian.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

// Helper function for Gaussian unit tests
static void testGaussian(const Gaussian<double> & p, const Eigen::VectorXd & mu_expected, const Eigen::MatrixXd & P_expected, const Eigen::VectorXd & eta_expected, const Eigen::MatrixXd & Lambda_expected);

SCENARIO("Gaussian construction")
{
    GIVEN("A Gaussian density")
    {
        const std::size_t n = 5;
        Eigen::MatrixXd S(n, n);
        S <<
            10, 11, 12, 13, 14,
             0, 15, 16, 17, 18,
             0,  0, 19, 20, 21,
             0,  0,  0, 22, 23,
             0,  0,  0,  0, 24;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");
        Eigen::MatrixXd P = S.transpose()*S;
        // Xi = qr(S^{-T})
        Eigen::MatrixXd Xi = S.triangularView<Eigen::Upper>().transpose().solve(
            Eigen::MatrixXd::Identity(S.rows(), S.cols())
        );
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(Xi);               // In-place QR decomposition
        Xi = Xi.triangularView<Eigen::Upper>();                                 // Safe aliasing
        Eigen::MatrixXd Lambda = Xi.transpose()*Xi;

        GIVEN("Non-zero mean")
        {
            Eigen::VectorXd mu(n);
            mu << 1, 2, 3, 4, 5;
            Eigen::VectorXd nu = Xi*mu;
            Eigen::VectorXd eta = Lambda*mu;

            WHEN("Creating a Gaussian from mean and sqrt covariance")
            {
                auto p = Gaussian<double>::fromSqrtMoment(mu, S);
                testGaussian(p, mu, P, eta, Lambda);
            }

            WHEN("Creating a Gaussian from mean and covariance")
            {
                auto p = Gaussian<double>::fromMoment(mu, P);
                testGaussian(p, mu, P, eta, Lambda);
            }

            WHEN("Creating a Gaussian from information vector and information matrix")
            {
                auto p = Gaussian<double>::fromInfo(eta, Lambda);
                testGaussian(p, mu, P, eta, Lambda);
            }

            WHEN("Creating a Gaussian from sqrt information vector and sqrt information matrix")
            {
                auto p = Gaussian<double>::fromSqrtInfo(nu, Xi);
                testGaussian(p, mu, P, eta, Lambda);
            }
        }

        GIVEN("Zero mean")
        {
            Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
            Eigen::VectorXd eta = Eigen::VectorXd::Zero(n);

            WHEN("Creating a Gaussian from sqrt covariance")
            {
                auto p = Gaussian<double>::fromSqrtMoment(S);
                testGaussian(p, mu, P, eta, Lambda);
            }

            WHEN("Creating a Gaussian from covariance")
            {
                auto p = Gaussian<double>::fromMoment(P);
                testGaussian(p, mu, P, eta, Lambda);
            }

            WHEN("Creating a Gaussian from information matrix")
            {
                auto p = Gaussian<double>::fromInfo(Lambda);
                testGaussian(p, mu, P, eta, Lambda);
            }

            WHEN("Creating a Gaussian from sqrt information matrix")
            {
                auto p = Gaussian<double>::fromSqrtInfo(Xi);
                testGaussian(p, mu, P, eta, Lambda);
            }
        }
    }
}

// Helper function for Gaussian density unit tests
void testGaussian(const Gaussian<double> & p, const Eigen::VectorXd & mu_expected, const Eigen::MatrixXd & P_expected, const Eigen::VectorXd & eta_expected, const Eigen::MatrixXd & Lambda_expected)
{
    const std::size_t n = mu_expected.size();
    REQUIRE(p.dim() == n);

    AND_WHEN("Evaluating mu = p.mean()")
    {
        Eigen::VectorXd mu = p.mean();

        THEN("mu has expected dimensions")
        {
            REQUIRE(mu.rows() == n);
            REQUIRE(mu.cols() == 1);

            AND_THEN("mu matches expected values")
            {
                CAPTURE_EIGEN(mu);
                CAPTURE_EIGEN(mu_expected);
                CHECK(mu.isApprox(mu_expected));
            }
        }
    }

    AND_WHEN("Evaluating eta = p.infoVec()")
    {
        Eigen::VectorXd eta = p.infoVec();

        THEN("eta has expected dimensions")
        {
            REQUIRE(eta.rows() == n);
            REQUIRE(eta.cols() == 1);

            AND_THEN("eta matches expected values")
            {
                CAPTURE_EIGEN(eta);
                CAPTURE_EIGEN(eta_expected);
                CHECK(eta.isApprox(eta_expected));
            }
        }
    }

    AND_WHEN("Evaluating Xi = p.sqrtInfoMat()")
    {
        Eigen::MatrixXd Xi = p.sqrtInfoMat();

        THEN("Xi has expected dimensions")
        {
            REQUIRE(Xi.rows() == n);
            REQUIRE(Xi.cols() == n);

            AND_THEN("Xi is upper triangular")
            {
                CAPTURE_EIGEN(Xi);
                CHECK(Xi.isUpperTriangular());
            }
        }
    }

    AND_WHEN("Evaluating nu = p.sqrtInfoVec()")
    {
        Eigen::VectorXd nu = p.sqrtInfoVec();

        THEN("nu has expected dimensions")
        {
            REQUIRE(nu.rows() == n);
            REQUIRE(nu.cols() == 1);

            AND_WHEN("Evaluating Xi = p.sqrtInfoMat()")
            {
                Eigen::MatrixXd Xi = p.sqrtInfoMat();

                THEN("Xi has expected dimensions")
                {
                    REQUIRE(Xi.rows() == n);
                    REQUIRE(Xi.cols() == n);
                }
            }
        }
    }

    AND_WHEN("Evaluating P = p.cov()")
    {
        Eigen::MatrixXd P = p.cov();

        THEN("P has expected dimensions")
        {
            REQUIRE(P.rows() == n);
            REQUIRE(P.cols() == n);

            AND_THEN("P matches expected values")
            {
                CAPTURE_EIGEN(P);
                CAPTURE_EIGEN(P_expected);
                CHECK(P.isApprox(P_expected));
            }
        }
    }

    AND_WHEN("Evaluating S = p.sqrtCov()")
    {
        Eigen::MatrixXd S = p.sqrtCov();

        THEN("S has expected dimensions")
        {
            REQUIRE(S.rows() == n);
            REQUIRE(S.cols() == n);

            AND_THEN("S is upper triangular")
            {
                CAPTURE_EIGEN(S);
                CHECK(S.isUpperTriangular());
            }
        }
    }

    AND_WHEN("Evaluating Lambda = p.infoMat()")
    {
        Eigen::MatrixXd Lambda = p.infoMat();

        THEN("Lambda has expected dimensions")
        {
            REQUIRE(Lambda.rows() == n);
            REQUIRE(Lambda.cols() == n);

            AND_THEN("Lambda matches expected values")
            {
                CAPTURE_EIGEN(Lambda);
                CAPTURE_EIGEN(Lambda_expected);
                CHECK(Lambda.isApprox(Lambda_expected));
            }
        }
    }

    AND_WHEN("Evaluating S = p.sqrtCov(), P = p.cov()")
    {
        Eigen::MatrixXd S = p.sqrtCov();
        Eigen::MatrixXd P = p.cov();

        THEN("S^T*S = P")
        {
            REQUIRE(S.cols() == P.rows());
            REQUIRE(S.cols() == P.cols());
            Eigen::MatrixXd LHS = S.transpose()*S;
            Eigen::MatrixXd RHS = P;
            CAPTURE_EIGEN(LHS);
            CAPTURE_EIGEN(RHS);
            CHECK(LHS.isApprox(RHS));
        }
    }

    AND_WHEN("Evaluating Xi = p.sqrtInfoMat(), Lambda = p.infoMat()")
    {
        Eigen::MatrixXd Xi = p.sqrtInfoMat();
        Eigen::MatrixXd Lambda = p.infoMat();

        THEN("Xi^T*Xi = Lambda")
        {
            REQUIRE(Xi.cols() == Lambda.rows());
            REQUIRE(Xi.cols() == Lambda.cols());
            Eigen::MatrixXd LHS = Xi.transpose()*Xi;
            Eigen::MatrixXd RHS = Lambda;
            CAPTURE_EIGEN(LHS);
            CAPTURE_EIGEN(RHS);
            CHECK(LHS.isApprox(RHS));
        }
    }

    AND_WHEN("Evaluating Lambda = p.infoMat(), mu = p.mean(), eta = p.infoVec()")
    {
        Eigen::MatrixXd Lambda = p.infoMat();
        Eigen::VectorXd mu = p.mean();
        Eigen::VectorXd eta = p.infoVec();

        THEN("Lambda*mu = eta")
        {
            REQUIRE(Lambda.cols() == mu.rows());
            REQUIRE(Lambda.rows() == eta.rows());
            REQUIRE(mu.cols() == eta.cols());
            Eigen::MatrixXd LHS = Lambda*mu;
            Eigen::MatrixXd RHS = eta;
            CAPTURE_EIGEN(LHS);
            CAPTURE_EIGEN(RHS);
            CHECK(LHS.isApprox(RHS));
        }
    }

    AND_WHEN("Evaluating P = p.cov(), eta = p.infoVec(), mu = p.mean()")
    {
        Eigen::MatrixXd P = p.cov();
        Eigen::VectorXd eta = p.infoVec();
        Eigen::VectorXd mu = p.mean();

        THEN("P*eta = mu")
        {
            REQUIRE(P.cols() == eta.rows());
            REQUIRE(P.rows() == mu.rows());
            REQUIRE(eta.cols() == mu.cols());
            Eigen::MatrixXd LHS = P*eta;
            Eigen::MatrixXd RHS = mu;
            CAPTURE_EIGEN(LHS);
            CAPTURE_EIGEN(RHS);
            CHECK(LHS.isApprox(RHS));
        }
    }

    AND_WHEN("Evaluating Xi = p.sqrtInfoMat(), mu = p.mean(), nu = p.sqrtInfoVec()")
    {
        Eigen::MatrixXd Xi = p.sqrtInfoMat();
        Eigen::VectorXd mu = p.mean();
        Eigen::VectorXd nu = p.sqrtInfoVec();

        THEN("Xi*mu = nu")
        {
            REQUIRE(Xi.cols() == mu.rows());
            REQUIRE(Xi.rows() == nu.rows());
            REQUIRE(mu.cols() == nu.cols());
            Eigen::MatrixXd LHS = Xi*mu;
            Eigen::MatrixXd RHS = nu;
            CAPTURE_EIGEN(LHS);
            CAPTURE_EIGEN(RHS);
            CHECK(LHS.isApprox(RHS));
        }
    }

    AND_WHEN("Evaluating Xi = p.sqrtInfoMat(), nu = p.sqrtInfoVec(), eta = p.infoVec()")
    {
        Eigen::MatrixXd Xi = p.sqrtInfoMat();
        Eigen::VectorXd nu = p.sqrtInfoVec();
        Eigen::VectorXd eta = p.infoVec();

        THEN("Xi^T*nu = eta")
        {
            REQUIRE(Xi.rows() == nu.rows());
            REQUIRE(Xi.cols() == eta.rows());
            REQUIRE(nu.cols() == eta.cols());
            Eigen::MatrixXd LHS = Xi.transpose()*nu;
            Eigen::MatrixXd RHS = eta;
            CAPTURE_EIGEN(LHS);
            CAPTURE_EIGEN(RHS);
            CHECK(LHS.isApprox(RHS));
        }
    }
}
