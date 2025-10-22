#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

// Example functions to test Gaussian transforms
static Eigen::VectorXd transformTestFuncSquareFullRank(const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::VectorXd f(2);
    double r = std::hypot(x(0), x(1));
    f(0) = std::atan2(x(0), x(1));
    f(1) = r;

    J.resize(2, 2);
    double r2 = r*r;
    J << x(1)/r2, -x(0)/r2,
         x(0)/r,   x(1)/r;

    return f;
}

static Eigen::VectorXd transformTestFuncWide(const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::VectorXd f(2);
    double r = std::hypot(x(0), x(1));
    f(0) = std::atan2(x(0), x(1));
    f(1) = r;

    J.resize(2, 3);
    double r2 = r*r;
    J << x(1)/r2, -x(0)/r2, 0,
         x(0)/r,   x(1)/r,  0;

    return f;
}

static Eigen::VectorXd transformTestFuncWideLowRank(const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::VectorXd f(2);
    double r = std::hypot(x(0), x(1));
    f(0) = 2*r;
    f(1) = r;

    J.resize(2, 3);
    J << 2*x(0)/r, 2*x(1)/r, 0,
         x(0)/r,   x(1)/r,   0;

    return f;
}

static Eigen::VectorXd transformTestFuncWide1Row(const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::VectorXd f(1);
    f(0) = std::hypot(x(0), x(1));

    J.resize(1, 2);
    J << x(0)/f(0), x(1)/f(0);

    return f;
}

static Eigen::VectorXd transformTestFuncTall(const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::VectorXd f(3);
    double r = std::hypot(x(0), x(1));
    f(0) = std::atan2(x(0), x(1));
    f(1) = r;
    f(2) = 2*r;

    J.resize(3, 2);
    double r2 = r*r;
    J << x(1)/r2, -x(0)/r2,
         x(0)/r,   x(1)/r,
         2*x(0)/r, 2*x(1)/r;

    return f;
}

static Eigen::VectorXd transformTestFuncSquareLowRank(const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::VectorXd f(2);
    double r = std::hypot(x(0), x(1));
    f(0) = r;
    f(1) = 2*r;

    J.resize(2, 2);
    J << x(0)/r,   x(1)/r,
         2*x(0)/r, 2*x(1)/r;

    return f;
}

SCENARIO("GaussianInfo affine transform")
{
    GIVEN("A bivariate Gaussian density")
    {
        Eigen::VectorXd mux2(2);
        Eigen::MatrixXd Sxx2(2,2);
        mux2 << 1, 3;
        Sxx2 << 0.1, 0.01,
                0,   0.01;
        auto px2 = GaussianInfo<double>::fromSqrtMoment(mux2, Sxx2);

        WHEN("Transforming through square full rank function")
        {
            auto py = px2.affineTransform(transformTestFuncSquareFullRank);

            Eigen::VectorXd muy = py.mean();
            Eigen::MatrixXd Pyy = py.cov();
            REQUIRE(muy.size() == 2);
            REQUIRE(Pyy.rows() == 2);
            REQUIRE(Pyy.cols() == 2);

            THEN("Mean matches expected values")
            {
                Eigen::VectorXd muy_exp(2);
                muy_exp << 0.321750554396642, 3.16227766016838;
                CAPTURE_EIGEN(muy);
                CAPTURE_EIGEN(muy_exp);
                CHECK(muy.isApprox(muy_exp, 1e-10));
            }

            THEN("Covariance matches expected values")
            {
                Eigen::MatrixXd Pyy_exp(2, 2);
                Pyy_exp << 0.000842, 0.00118269184490297,
                           0.00118269184490297, 0.00178;
                CAPTURE_EIGEN(Pyy);
                CAPTURE_EIGEN(Pyy_exp);
                CHECK(Pyy.isApprox(Pyy_exp, 1e-10));
            }
        }

        WHEN("Transforming through wide 1-row function")
        {
            auto py = px2.affineTransform(transformTestFuncWide1Row);

            double muy = py.mean()(0);
            double Pyy = py.cov()(0,0);

            THEN("Mean matches expected value")
            {
                double muy_exp = 3.16227766016838;
                CHECK(muy == doctest::Approx(muy_exp).epsilon(1e-10));
            }

            THEN("Variance matches expected value")
            {
                double Pyy_exp = 0.00178;
                CHECK(Pyy == doctest::Approx(Pyy_exp).epsilon(1e-10));
            }
        }

        WHEN("Transforming through tall function")
        {
            auto py = px2.affineTransform(transformTestFuncTall).marginal(Eigen::seq(0, 1));

            Eigen::VectorXd muy = py.mean();
            Eigen::MatrixXd Pyy = py.cov();
            REQUIRE(muy.size() == 2);
            REQUIRE(Pyy.rows() == 2);
            REQUIRE(Pyy.cols() == 2);

            THEN("Mean matches expected values")
            {
                Eigen::VectorXd muy_exp(2);
                muy_exp << 0.321750554396642, 3.16227766016838;
                CAPTURE_EIGEN(muy);
                CAPTURE_EIGEN(muy_exp);
                CHECK(muy.isApprox(muy_exp, 1e-7));
            }

            THEN("Covariance matches expected values")
            {
                Eigen::MatrixXd Pyy_exp(2, 2);
                Pyy_exp << 0.000842, 0.00118269184490297,
                           0.00118269184490297, 0.00178;
                CAPTURE_EIGEN(Pyy);
                CAPTURE_EIGEN(Pyy_exp);
                CHECK(Pyy.isApprox(Pyy_exp, 1e-7));
            }
        }

        WHEN("Transforming through square low rank function")
        {
            auto py = px2.affineTransform(transformTestFuncSquareLowRank);

            Eigen::VectorXd muy = py.mean();
            Eigen::MatrixXd Pyy = py.cov();
            REQUIRE(muy.size() == 2);
            REQUIRE(Pyy.rows() == 2);
            REQUIRE(Pyy.cols() == 2);

            THEN("Mean matches expected values")
            {
                Eigen::VectorXd muy_exp(2);
                muy_exp << 3.16227766016838, 6.32455531152522;
                CAPTURE_EIGEN(muy);
                CAPTURE_EIGEN(muy_exp);
                CHECK(muy.isApprox(muy_exp, 1e-7));
            }

            THEN("Covariance matches expected values")
            {
                Eigen::MatrixXd Pyy_exp(2, 2);
                Pyy_exp << 0.00178, 0.00356,
                           0.00356, 0.00712;
                CAPTURE_EIGEN(Pyy);
                CAPTURE_EIGEN(Pyy_exp);
                CHECK(Pyy.isApprox(Pyy_exp, 1e-7));
            }
        }
    }

    GIVEN("A trivariate Gaussian density")
    {
        Eigen::VectorXd mux3(3);
        Eigen::MatrixXd Sxx3(3,3);
        mux3 << 1, 3, 5;
        Sxx3 << 0.1, 0.01, 0,
                0,   0.01, 0,
                0,   0,    1;
        auto px3 = GaussianInfo<double>::fromSqrtMoment(mux3, Sxx3);

        WHEN("Transforming through wide function")
        {
            auto py = px3.affineTransform(transformTestFuncWide);

            Eigen::VectorXd muy = py.mean();
            Eigen::MatrixXd Pyy = py.cov();
            REQUIRE(muy.size() == 2);
            REQUIRE(Pyy.rows() == 2);
            REQUIRE(Pyy.cols() == 2);

            THEN("Mean matches expected values")
            {
                Eigen::VectorXd muy_exp(2);
                muy_exp << 0.321750554396642, 3.16227766016838;
                CAPTURE_EIGEN(muy);
                CAPTURE_EIGEN(muy_exp);
                CHECK(muy.isApprox(muy_exp, 1e-10));
            }

            THEN("Covariance matches expected values")
            {
                Eigen::MatrixXd Pyy_exp(2, 2);
                Pyy_exp << 0.000842, 0.00118269184490297,
                           0.00118269184490297, 0.00178;
                CAPTURE_EIGEN(Pyy);
                CAPTURE_EIGEN(Pyy_exp);
                CHECK(Pyy.isApprox(Pyy_exp, 1e-10));
            }
        }

        WHEN("Transforming through wide low-rank function")
        {
            auto py = px3.affineTransform(transformTestFuncWideLowRank).marginal(Eigen::seq(1, 1));

            double muy = py.mean()(0);
            double Pyy = py.cov()(0,0);

            THEN("Mean matches expected value")
            {
                double muy_exp = 3.16227766016838;
                CHECK(muy == doctest::Approx(muy_exp).epsilon(1e-7));
            }

            THEN("Variance matches expected value")
            {
                double Pyy_exp = 0.00178;
                CHECK(Pyy == doctest::Approx(Pyy_exp).epsilon(1e-7));
            }
        }
    }
}
