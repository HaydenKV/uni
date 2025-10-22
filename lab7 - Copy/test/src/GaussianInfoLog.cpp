#include <doctest/doctest.h>
#include <cmath>
#include <numbers>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "../../src/GaussianInfo.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("GaussianInfo log nominal")
{
    GIVEN("Nominal parameters")
    {
        Eigen::VectorXd x(3);
        x << 1, 2, 3;
        Eigen::VectorXd mu(3);
        mu << 2, 4, 6;
        Eigen::MatrixXd S(3, 3);
        S << -1, -4, -6,
              0, -2, -5,
              0,  0, -3;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        
        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-5.7707972911));
            }
        }

        WHEN("Evaluating l = p.log(x, g)")
        {
            Eigen::VectorXd g;
            double l = p.log(x, g);

            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-5.7707972911));
            }

            AND_WHEN("Computing autodiff gradient")
            {
                Eigen::VectorXd g_exp;
                GaussianInfo<autodiff::dual> pdual = p.cast<autodiff::dual>();
                Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
                autodiff::dual fdual;
                const auto func = [&](const Eigen::VectorX<autodiff::dual> & x) { return pdual.log(x); };
                g_exp = gradient(func, wrt(xdual), at(xdual), fdual);
                REQUIRE(g_exp.rows() == x.size());
                REQUIRE(g_exp.cols() == 1);

                THEN("l matches autodiff value")
                {
                    CHECK(l == doctest::Approx(val(fdual)));
                }

                THEN("g has expected dimensions")
                {
                    REQUIRE(g.rows() == x.size());
                    REQUIRE(g.cols() == 1);

                    AND_THEN("g matches autodiff gradient")
                    {
                        CAPTURE_EIGEN(g);
                        CAPTURE_EIGEN(g_exp);
                        CHECK(g.isApprox(g_exp));
                    }
                }
            }
        }

        WHEN("Evaluating l = p.log(x, g, H)")
        {
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            double l = p.log(x, g, H);

            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-5.7707972911));
            }

            AND_WHEN("Computing autodiff gradient and Hessian")
            {
                Eigen::VectorXd g_exp;
                Eigen::MatrixXd H_exp;
                GaussianInfo<autodiff::dual2nd> pdual = p.cast<autodiff::dual2nd>();
                Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
                autodiff::dual2nd fdual;
                const auto func = [&](const Eigen::VectorX<autodiff::dual2nd> & x) { return pdual.log(x); };
                H_exp = hessian(func, wrt(xdual), at(xdual), fdual, g_exp);
                REQUIRE(g_exp.rows() == x.size());
                REQUIRE(g_exp.cols() == 1);
                REQUIRE(H_exp.rows() == x.size());
                REQUIRE(H_exp.cols() == x.size());

                THEN("l matches autodiff value")
                {
                    CHECK(l == doctest::Approx(val(fdual)));
                }

                THEN("g has expected dimensions")
                {
                    REQUIRE(g.rows() == x.size());
                    REQUIRE(g.cols() == 1);

                    AND_THEN("g matches autodiff gradient")
                    {
                        CAPTURE_EIGEN(g);
                        CAPTURE_EIGEN(g_exp);
                        CHECK(g.isApprox(g_exp));
                    }
                }

                THEN("H has expected dimensions")
                {
                    REQUIRE(H.rows() == x.size());
                    REQUIRE(H.cols() == x.size());

                    AND_THEN("H matches autodiff Hessian")
                    {
                        CAPTURE_EIGEN(H);
                        CAPTURE_EIGEN(H_exp);
                        CHECK(H.isApprox(H_exp));
                    }
                }
            }
        }
    }
}

SCENARIO("Gaussian log exponential underflow")
{
    GIVEN("Parameters that may cause underflow in the exponential")
    {
        Eigen::VectorXd x(1);
        x << 0;
        Eigen::VectorXd mu(1);
        mu << std::sqrt(350*std::log(10)/std::numbers::pi); // Approx 16
        Eigen::MatrixXd S(1, 1);
        S << 1.0/std::sqrt(2*std::numbers::pi); // Approx 0.4
        REQUIRE( std::exp( -0.5*S.triangularView<Eigen::Upper>().transpose().solve(x - mu).squaredNorm() ) == 0.0);

        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        
        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-805.904782547916));
            }
        }
    }
}

#include <Eigen/LU> // For .determinant()

SCENARIO("Gaussian log determinant underflow")
{
    GIVEN("Parameters that may cause determinant underflow")
    {
        double a = 1e-4;    // Magnitude of st.dev.
        int n = 100;        // Dimension
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd S = a*Eigen::MatrixXd::Identity(n, n);
        REQUIRE(S.determinant() == 0.0); // underflow to zero

        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        
        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-n*std::log(a) - n/2.0*std::log(2*std::numbers::pi)));
            }
        }
    }
}

SCENARIO("Gaussian log determinant overflow")
{
    GIVEN("Parameters that may cause determinant overflow")
    {
        double a = 1e4;     // Magnitude of st.dev.
        int n = 100;        // Dimension
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd S = a*Eigen::MatrixXd::Identity(n, n);
        REQUIRE(S.determinant() == std::numeric_limits<double>::infinity()); // overflow to infinity
        
        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);

        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-n*std::log(a) - n/2.0*std::log(2*std::numbers::pi)));
            }
        }
    }
}

SCENARIO("Gaussian log covariance overflow")
{
    GIVEN("Parameters that may cause the covariance to overflow")
    {
        int n = 2;
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd S = 1e300*Eigen::MatrixXd::Identity(n, n);
        REQUIRE_FALSE((S.transpose()*S).array().isFinite().all());

        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);

        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l is finite")
            {
                CHECK(std::isfinite(l));
            }
        }
    }
}
