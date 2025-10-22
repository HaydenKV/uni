#include <doctest/doctest.h>
#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "../../src/Gaussian.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

// Helper functions for Gaussian conditional density unit tests
template <typename IndexTypeA, typename IndexTypeB>
static void testConditionalValue(const Gaussian<double> & p, const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorXd & xB)
{
    const std::size_t & nA = idxA.size();
    const std::size_t & nB = idxB.size();
    REQUIRE(nA + nB == p.dim());
    
    auto c = p.conditional(idxA, idxB, xB);
    REQUIRE(c.dim() == nA);
    REQUIRE(c.sqrtCov().rows() == nA);

    Eigen::VectorXd mu = p.mean();
    Eigen::MatrixXd P = p.cov();

    THEN("Conditional mean matches expected result")
    {
        Eigen::VectorXd muc = c.mean();
        REQUIRE(muc.rows() == nA);
        REQUIRE(muc.cols() == 1);
        Eigen::VectorXd muc_expected =
            mu(idxA) + P(idxA, idxB)*P(idxB, idxB).llt().solve(xB - mu(idxB));
        CAPTURE_EIGEN(muc);
        CAPTURE_EIGEN(muc_expected);
        CHECK(muc.isApprox(muc_expected));
    }

    THEN("Conditional sqrt cov is upper triangular")
    {
        Eigen::MatrixXd Sc = c.sqrtCov();
        CAPTURE_EIGEN(Sc);
        REQUIRE(Sc.isUpperTriangular());
    }

    THEN("Conditional cov matches expected result")
    {
        Eigen::MatrixXd Pc = c.cov();
        REQUIRE(Pc.rows() == nA);
        REQUIRE(Pc.cols() == nA);
        Eigen::MatrixXd Pc_expected =
            P(idxA, idxA) - P(idxA, idxB)*P(idxB, idxB).llt().solve(P(idxB, idxA));
        CAPTURE_EIGEN(Pc);
        CAPTURE_EIGEN(Pc_expected);
        CHECK(Pc.isApprox(Pc_expected));
    }
}

template <typename IndexTypeA, typename IndexTypeB>
static void testConditionalDist(const Gaussian<double> & p, const IndexTypeA & idxA, const IndexTypeB & idxB, const Gaussian<double> & pB)
{
    const std::size_t & nA = idxA.size();
    const std::size_t & nB = idxB.size();
    REQUIRE(nA + nB == p.dim());
    REQUIRE(pB.dim() == nB);
    
    auto c = p.conditional(idxA, idxB, pB);
    REQUIRE(c.dim() == nA);
    REQUIRE(c.sqrtCov().rows() == nA);

    Eigen::VectorXd mu = p.mean();
    Eigen::MatrixXd P = p.cov();

    THEN("Conditional mean matches expected result")
    {
        Eigen::VectorXd muc = c.mean();
        REQUIRE(muc.rows() == nA);
        REQUIRE(muc.cols() == 1);
        Eigen::VectorXd muc_expected =
            mu(idxA) + P(idxA, idxB)*P(idxB, idxB).llt().solve(pB.mean() - mu(idxB));
        CAPTURE_EIGEN(muc);
        CAPTURE_EIGEN(muc_expected);
        CHECK(muc.isApprox(muc_expected));
    }

    THEN("Conditional sqrt cov is upper triangular")
    {
        Eigen::MatrixXd Sc = c.sqrtCov();
        CAPTURE_EIGEN(Sc);
        REQUIRE(Sc.isUpperTriangular());
    }

    THEN("Conditional cov matches expected result")
    {
        Eigen::MatrixXd Pc = c.cov();
        REQUIRE(Pc.rows() == nA);
        REQUIRE(Pc.cols() == nA);
        Eigen::MatrixXd PBA = P(idxB, idxB).llt().solve(P(idxB, idxA)); // PBB\PBA
        Eigen::MatrixXd Pc_expected =
            P(idxA, idxA) + PBA.transpose()*(pB.cov() - P(idxB, idxB))*PBA;
        CAPTURE_EIGEN(Pc);
        CAPTURE_EIGEN(Pc_expected);
        CHECK(Pc.isApprox(Pc_expected));
    }
}

template <typename IndexTypeA, typename IndexTypeB>
static void testConditional(const Gaussian<double> & p, const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorXd & xB)
{
    // Condition on a value
    testConditionalValue(p, idxA, idxB, xB);

    // Condition on a delta distribution
    Gaussian<double> pB_delta = Gaussian<double>::fromSqrtMoment(xB, Eigen::MatrixXd::Zero(xB.size(), xB.size()));
    testConditionalDist(p, idxA, idxB, pB_delta);

    // Condition on a Gaussian distribution
    Gaussian<double> pB = Gaussian<double>::fromSqrtMoment(xB, Eigen::MatrixXd::Identity(xB.size(), xB.size()));
    testConditionalDist(p, idxA, idxB, pB);
}

SCENARIO("Gaussian conditional density")
{
    GIVEN("A joint Gaussian density (size = 4)")
    {
        const std::size_t n = 4;
        Eigen::VectorXd mu(n);
        mu <<                  1,
                               1,
                               1,
                               1;
        Eigen::MatrixXd S(n, n);
        S <<  -0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069,
                               0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469,
                               0,                   0,  -0.196861446475943,    0.800320709801823,
                               0,                   0,                   0,    -1.50940472473439;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        auto p = Gaussian<double>::fromSqrtMoment(mu, S);
        REQUIRE(p.dim() == n);

        GIVEN("A vector (size = 3)")
        {
            Eigen::VectorXd xB(3);
            xB << 0.875874147834533,
                  -0.24278953633334,
                  0.166813439453503;

            WHEN("Conditioning on head")
            {
                std::vector<int> idxA = {3};
                std::vector<int> idxB = {0, 1, 2};
                testConditional(p, idxA, idxB, xB);
            }

            WHEN("Conditioning on tail")
            {
                std::vector<int> idxA = {0};
                std::vector<int> idxB = {1, 2, 3};
                testConditional(p, idxA, idxB, xB);
            }
        }

        GIVEN("A vector (size = 1)")
        {
            Eigen::VectorXd xB(1);
            xB << 0.875874147834533;

            WHEN("Conditioning on head")
            {
                std::vector<int> idxA = {1, 2, 3};
                std::vector<int> idxB = {0};
                testConditional(p, idxA, idxB, xB);
            }

            WHEN("Conditioning on tail")
            {
                std::vector<int> idxA = {0, 1, 2};
                std::vector<int> idxB = {3};
                testConditional(p, idxA, idxB, xB);
            }

            WHEN("Conditioning on segment")
            {
                std::vector<int> idxA = {0, 1, 3};
                std::vector<int> idxB = {2};
                testConditional(p, idxA, idxB, xB);
            }
        }
    }

    GIVEN("A joint Gaussian density (size = 6)")
    {
        const std::size_t n = 6;
        Eigen::VectorXd mu(n);
        mu <<                  1,
                               1,
                               1,
                               1,
                               1,
                               1;
        Eigen::MatrixXd S(n, n);
        S <<  -0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069,   -1.50940472473439,   0.166813439453503,
                               0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469,   0.875874147834533,   -1.96541870928278,
                               0,                   0,  -0.196861446475943,    0.800320709801823,   -0.24278953633334,   -1.27007139263854,
                               0,                   0,                   0,     1.17517126546302,   0.603658445825815,   -1.86512257453063,
                               0,                   0,                   0,                    0,     1.7812518932425,   -1.05110705924059,
                               0,                   0,                   0,                    0,                   0,  -0.417382047996795;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        auto p = Gaussian<double>::fromSqrtMoment(mu, S);
        REQUIRE(p.dim() == n);

        GIVEN("A vector (size = 3)")
        {
            Eigen::VectorXd xB(3);
            xB << 0.875874147834533,
                  -0.24278953633334,
                  0.166813439453503;

            WHEN("Conditioning on head")
            {
                std::vector<int> idxA = {3, 4, 5};
                std::vector<int> idxB = {0, 1, 2};
                testConditional(p, idxA, idxB, xB);
            }

            WHEN("Conditioning on tail")
            {
                std::vector<int> idxA = {0, 1, 2};
                std::vector<int> idxB = {3, 4, 5};
                testConditional(p, idxA, idxB, xB);
            }

            WHEN("Conditioning on segment")
            {
                std::vector<int> idxA = {0, 1, 5};
                std::vector<int> idxB = {2, 3, 4};
                testConditional(p, idxA, idxB, xB);
            }

            WHEN("Conditioning on non-continguous elements")
            {
                std::vector<int> idxA = {0, 2, 4};
                std::vector<int> idxB = {1, 3, 5};
                testConditional(p, idxA, idxB, xB);
            }

            WHEN("Conditioning on non-continguous elements in non-ascending order")
            {
                std::vector<int> idxA = {2, 0, 4};
                std::vector<int> idxB = {1, 5, 3};
                testConditional(p, idxA, idxB, xB);
            }
        }
    }
}

SCENARIO("Gaussian conditional covariance overflow")
{
    GIVEN("Parameters that may cause the covariance to overflow")
    {
        int n = 2;
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd S = 1e300*Eigen::MatrixXd::Identity(n, n);
        auto p = Gaussian<double>::fromSqrtMoment(mu, S);
        // Make sure that covariance overflows
        REQUIRE_FALSE(p.cov().array().isFinite().all());

        WHEN("Evaluating pc = p.conditional(1, 0, 0)")
        {
            std::vector<int> idxA = {1};
            std::vector<int> idxB = {0};
            Eigen::VectorXd xB(1);
            xB.setZero();
            auto pc = p.conditional(idxA, idxB, xB);
            THEN("pc.sqrtCov() is finite")
            {
                Eigen::MatrixXd Sc = pc.sqrtCov();
                CAPTURE_EIGEN(Sc);
                CHECK(Sc.array().isFinite().all());
            }
        }
    }
}
