#include <doctest/doctest.h>
#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "../../src/Gaussian.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

// Helper function for Gaussian marginal density unit tests
template <typename IndexType>
static void testMarginal(const Gaussian<double> & p, const IndexType & idx)
{
    const std::size_t n = idx.size();
    auto m = p.marginal(idx);
    REQUIRE(m.dim() == n);
    REQUIRE(m.sqrtCov().rows() == n);

    THEN("Marginal mean matches expected result")
    {
        Eigen::VectorXd mum = m.mean();
        REQUIRE(mum.rows() == n);
        REQUIRE(mum.cols() == 1);
        Eigen::VectorXd mu = p.mean();
        Eigen::VectorXd mum_expected = mu(idx);
        CAPTURE_EIGEN(mum);
        CAPTURE_EIGEN(mum_expected);
        CHECK(mum.isApprox(mum_expected));
    }

    THEN("Marginal sqrt cov is upper triangular")
    {
        Eigen::MatrixXd Sm = m.sqrtCov();
        CAPTURE_EIGEN(Sm);
        CHECK(Sm.isUpperTriangular());
    }

    THEN("Marginal cov matches expected result")
    {
        Eigen::MatrixXd Pm = m.cov();
        REQUIRE(Pm.rows() == n);
        REQUIRE(Pm.cols() == n);
        Eigen::MatrixXd P = p.cov();
        Eigen::MatrixXd Pm_expected = P(idx, idx);
        CAPTURE_EIGEN(Pm);
        CAPTURE_EIGEN(Pm_expected);
        CHECK(Pm.isApprox(Pm_expected));
    }
}

SCENARIO("Gaussian marginal density")
{
    GIVEN("A Gaussian density (size = 5)")
    {
        Eigen::VectorXd mu(5);
        mu << 1, 2, 3, 4, 5;
        Eigen::MatrixXd S(5, 5);
        S <<
            10, 11, 12, 13, 14,
             0, 15, 16, 17, 18,
             0,  0, 19, 20, 21,
             0,  0,  0, 22, 23,
             0,  0,  0,  0, 24;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        auto p = Gaussian<double>::fromSqrtMoment(mu, S);
        REQUIRE(p.dim() == 5);

        WHEN("Extracting marginal head (size = 2)")
        {
            std::vector<int> idx = {0, 1};
            testMarginal(p, idx);
        }

        WHEN("Extracting marginal tail (size = 2)")
        {
            std::vector<int> idx = {3, 4};
            testMarginal(p, idx);
        }

        WHEN("Extracting marginal segment (size = 3)")
        {
            std::vector<int> idx = {1, 2, 3};
            testMarginal(p, idx);
        }

        WHEN("Extracting marginal non-continguous elements (size = 3)")
        {
            std::vector<int> idx = {0, 2, 4};
            testMarginal(p, idx);
        }

        WHEN("Extracting marginal non-continguous elements in non-ascending order (size = 3)")
        {
            std::vector<int> idx = {4, 2, 0};
            testMarginal(p, idx);
        }
    }
}

SCENARIO("Gaussian marginal covariance overflow")
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

        WHEN("Evaluating pm = p.marginal({0, 1})")
        {
            std::vector<int> idx = {0, 1};
            auto pm = p.marginal(idx);
            THEN("pm.sqrtCov() is finite")
            {
                Eigen::MatrixXd Sm = pm.sqrtCov();
                CAPTURE_EIGEN(Sm);
                CHECK(Sm.array().isFinite().all());
            }
        }
    }
}
