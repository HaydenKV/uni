#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include "../../src/GaussianBase.hpp"
#include "../../src/GaussianInfo.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("GaussianInfo logIntegral")
{
    GIVEN("Mean within interval")
    {
        double mu = 2;
        double sigma = 1;
        double a = 0;
        double b = 3;
        REQUIRE(a <= mu);
        REQUIRE(mu <= b);
        auto p = GaussianInfo<double>::fromSqrtMoment(Eigen::Vector<double, 1>(mu), Eigen::Matrix<double, 1, 1>(sigma));

        WHEN("Evaluating l = p.logIntegral(a, b)")
        {
            double l = p.logIntegral(a, b);
            THEN("l matches expected value")
            {
                double expected = std::log(GaussianBase<double>::normcdf(b, mu, sigma) - GaussianBase<double>::normcdf(a, mu, sigma));
                CHECK(l == doctest::Approx(expected));
            }
        }

        WHEN("Evaluating l = p.logIntegral(a, b, g)")
        {
            double g;
            double l = p.logIntegral(a, b, g);

            THEN("l matches expected value")
            {
                double expected = std::log(GaussianBase<double>::normcdf(b, mu, sigma) - GaussianBase<double>::normcdf(a, mu, sigma));
                CHECK(l == doctest::Approx(expected));
            }

            THEN("g matches expected value")
            {
                auto p2 = GaussianInfo<double>::fromSqrtMoment(
                    Eigen::VectorXd::Constant(1, mu),
                    Eigen::MatrixXd::Constant(1, 1, sigma)
                );
                
                double pdf_a = p2.eval(Eigen::VectorXd::Constant(1, a)); // normpdf(a, mu, sigma)
                double pdf_b = p2.eval(Eigen::VectorXd::Constant(1, b)); // normpdf(b, mu, sigma)

                double cdf_a = GaussianBase<double>::normcdf(a, mu, sigma);
                double cdf_b = GaussianBase<double>::normcdf(b, mu, sigma);
                
                // The log integral is ln(Φ(b) - Φ(a)), where Φ is the cumulative distribution function (CDF) of the normal distribution.
                // The derivative of this with respect to μ is: (φ(a) - φ(b)) / (Φ(b) - Φ(a)) where φ is the probability density function (PDF) of the normal distribution.
                double g_exp = (pdf_a - pdf_b) / (cdf_b - cdf_a);
                CHECK(g == doctest::Approx(g_exp));
            }
        }
    }

    GIVEN("Mean below interval")
    {
        double mu = -1;
        double sigma = 1;
        double a = 0;
        double b = 3;
        REQUIRE(mu < a);
        auto p = GaussianInfo<double>::fromSqrtMoment(Eigen::Vector<double, 1>(mu), Eigen::Matrix<double, 1, 1>(sigma));

        WHEN("Evaluating l = p.logIntegral(a, b)")
        {
            double l = p.logIntegral(a, b);
            THEN("l matches expected value")
            {
                double expected = std::log(GaussianBase<double>::normcdf(b, mu, sigma) - GaussianBase<double>::normcdf(a, mu, sigma));
                CHECK(l == doctest::Approx(expected));
            }
        }

        WHEN("Evaluating l = p.logIntegral(a, b, g)")
        {
            double g;
            double l = p.logIntegral(a, b, g);

            THEN("l matches expected value")
            {
                double expected = std::log(GaussianBase<double>::normcdf(b, mu, sigma) - GaussianBase<double>::normcdf(a, mu, sigma));
                CHECK(l == doctest::Approx(expected));
            }

            THEN("g matches expected value")
            {
                auto p2 = GaussianInfo<double>::fromSqrtMoment(
                    Eigen::VectorXd::Constant(1, mu),
                    Eigen::MatrixXd::Constant(1, 1, sigma)
                );

                double pdf_a = p2.eval(Eigen::VectorXd::Constant(1, a)); // normpdf(a, mu, sigma)
                double pdf_b = p2.eval(Eigen::VectorXd::Constant(1, b)); // normpdf(b, mu, sigma)

                double cdf_a = GaussianBase<double>::normcdf(a, mu, sigma);
                double cdf_b = GaussianBase<double>::normcdf(b, mu, sigma);
                
                // The log integral is ln(Φ(b) - Φ(a)), where Φ is the cumulative distribution function (CDF) of the normal distribution.
                // The derivative of this with respect to μ is: (φ(a) - φ(b)) / (Φ(b) - Φ(a)) where φ is the probability density function (PDF) of the normal distribution.
                double g_exp = (pdf_a - pdf_b) / (cdf_b - cdf_a);
                CHECK(g == doctest::Approx(g_exp));
            }
        }
    }

    GIVEN("Mean above interval")
    {
        double mu = 4;
        double sigma = 1;
        double a = 0;
        double b = 3;
        REQUIRE(mu > b);
        auto p = GaussianInfo<double>::fromSqrtMoment(Eigen::Vector<double, 1>(mu), Eigen::Matrix<double, 1, 1>(sigma));

        WHEN("Evaluating l = p.logIntegral(a, b)")
        {
            double l = p.logIntegral(a, b);
            THEN("l matches expected value")
            {
                double expected = std::log(GaussianBase<double>::normcdf(b, mu, sigma) - GaussianBase<double>::normcdf(a, mu, sigma));
                CHECK(l == doctest::Approx(expected));
            }
        }

        WHEN("Evaluating l = p.logIntegral(a, b, g)")
        {
            double g;
            double l = p.logIntegral(a, b, g);

            THEN("l matches expected value")
            {
                double expected = std::log(GaussianBase<double>::normcdf(b, mu, sigma) - GaussianBase<double>::normcdf(a, mu, sigma));
                CHECK(l == doctest::Approx(expected));
            }

            THEN("g matches expected value")
            {
                auto p2 = GaussianInfo<double>::fromSqrtMoment(
                    Eigen::VectorXd::Constant(1, mu),
                    Eigen::MatrixXd::Constant(1, 1, sigma)
                );
                
                double pdf_a = p2.eval(Eigen::VectorXd::Constant(1, a)); // normpdf(a, mu, sigma)
                double pdf_b = p2.eval(Eigen::VectorXd::Constant(1, b)); // normpdf(b, mu, sigma)

                double cdf_a = GaussianBase<double>::normcdf(a, mu, sigma);
                double cdf_b = GaussianBase<double>::normcdf(b, mu, sigma);
                
                // The log integral is ln(Φ(b) - Φ(a)), where Φ is the cumulative distribution function (CDF) of the normal distribution.
                // The derivative of this with respect to μ is: (φ(a) - φ(b)) / (Φ(b) - Φ(a)) where φ is the probability density function (PDF) of the normal distribution.
                double g_exp = (pdf_a - pdf_b) / (cdf_b - cdf_a);
                CHECK(g == doctest::Approx(g_exp));
            }
        }
    }
}
