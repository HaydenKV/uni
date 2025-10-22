#include <doctest/doctest.h>
#include <cstddef>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../src/Gaussian.hpp"
#include "../../src/SystemBallistic.h"
#include "../../src/MeasurementRADAR.h"


#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("MeasurementRADAR")
{
    const Eigen::Index ny = 1;
    Eigen::VectorXd y(ny);
    y << 100.0;
    double t = 0.0;
    MeasurementRADAR measurement(t, y);
    Gaussian<double> p0 = Gaussian<double>::fromSqrtMoment(Eigen::MatrixXd::Zero(3, 3));
    SystemBallistic system(p0);

    GIVEN("A state vector")
    {
        const Eigen::Index nx = 3;
        Eigen::VectorXd x(nx);
        x <<    13955,
             -449.745,
               0.0005;

        WHEN("Evaluating h = measurement.predict(x)")
        {
            Eigen::VectorXd h = measurement.predict(x, system);

            THEN("h has expected dimensions")
            {
                REQUIRE(h.rows() == ny);
                REQUIRE(h.cols() == 1);

                AND_THEN("h has expected values")
                {
                    Eigen::VectorXd h_expected(ny);
                    h_expected << 10256.3163465252;
                    CAPTURE_EIGEN(h);
                    CAPTURE_EIGEN(h_expected);
                    CHECK(h.isApprox(h_expected));
                }
            }
        }

        WHEN("Evaluating h = measurement.predict(x, J)")
        {
            Eigen::MatrixXd J;
            Eigen::VectorXd h = measurement.predict(x, system, J);

            THEN("h has expected dimensions")
            {
                REQUIRE(h.rows() == ny);
                REQUIRE(h.cols() == 1);

                AND_THEN("h has expected values")
                {
                    Eigen::VectorXd h_expected(ny);
                    h_expected << 10256.3163465252;
                    CAPTURE_EIGEN(h);
                    CAPTURE_EIGEN(h_expected);
                    CHECK(h.isApprox(h_expected));
                }
            }

            THEN("J has expected dimensions")
            {
                REQUIRE(J.rows() == ny);
                REQUIRE(J.cols() == nx);

                AND_THEN("J has expected values")
                {
                    Eigen::MatrixXd J_expected(ny, nx);
                    J_expected << 0.873120494477915, 0, 0;
                    CAPTURE_EIGEN(J);
                    CAPTURE_EIGEN(J_expected);
                    CHECK(J.isApprox(J_expected));
                }
            }
        }

        WHEN("Evaluating h = measurement.predict(x, J, H)")
        {
            Eigen::MatrixXd J;
            Eigen::Tensor<double, 3> H;
            Eigen::VectorXd h = measurement.predict(x, system, J, H);

            THEN("h has expected dimensions")
            {
                REQUIRE(h.rows() == ny);
                REQUIRE(h.cols() == 1);

                AND_THEN("h has expected values")
                {
                    Eigen::VectorXd h_expected(ny);
                    h_expected << 10256.3163465252;
                    CAPTURE_EIGEN(h);
                    CAPTURE_EIGEN(h_expected);
                    CHECK(h.isApprox(h_expected));
                }
            }

            THEN("J has expected dimensions")
            {
                REQUIRE(J.rows() == ny);
                REQUIRE(J.cols() == nx);

                AND_THEN("J has expected values")
                {
                    Eigen::MatrixXd J_expected(ny, nx);
                    J_expected << 0.873120494477915, 0, 0;
                    CAPTURE_EIGEN(J);
                    CAPTURE_EIGEN(J_expected);
                    CHECK(J.isApprox(J_expected));
                }
            }

            THEN("H has expected dimensions")
            {
                REQUIRE(H.dimension(0) == ny);
                REQUIRE(H.dimension(1) == nx);
                REQUIRE(H.dimension(2) == nx);

                AND_THEN("H has expected values")
                {
                    Eigen::Tensor<double, 3> H_exp(ny, nx, nx);
                    H_exp.setValues({ { {2.31721e-05, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} } });
                    for (Eigen::Index i = 0; i < ny; ++i)
                    {
                        for (Eigen::Index j = 0; j < nx; ++j)
                        {
                            for (Eigen::Index k = 0; k < nx; ++k)
                            {
                                INFO("Tensor indices: i = ", i, ", j = ", j, ", k = ", k);
                                CHECK(H(i, j, k) == doctest::Approx(H_exp(i, j, k)));
                            }
                        }
                    }
                }
            }
        }

        WHEN("Evaluating h = measurement.logLikelihood(x)")
        {
            double logLik = measurement.logLikelihood(x, system);

            THEN("logLik has expected value")
            {
                CHECK(logLik == doctest::Approx(-20635.0));
            }
        }

        WHEN("Evaluating h = measurement.logLikelihood(x, g)")
        {
            Eigen::VectorXd g;
            double logLik = measurement.logLikelihood(x, system, g);

            THEN("logLik has expected value")
            {
                CHECK(logLik == doctest::Approx(-20635.0));
            }

            THEN("g has expected dimensions")
            {
                REQUIRE(g.rows() == nx);
                REQUIRE(g.cols() == 1);

                AND_THEN("g has expected values")
                {
                    Eigen::VectorXd g_exp(nx);
                    g_exp << -3.54707518022088, 0.0, 0.0;
                    CAPTURE_EIGEN(g);
                    CAPTURE_EIGEN(g_exp);
                    CHECK(g.isApprox(g_exp));
                }
            }
        }

        WHEN("Evaluating h = measurement.logLikelihood(x, g, H)")
        {
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            double logLik = measurement.logLikelihood(x, system, g, H);

            THEN("logLik has expected value")
            {
                CHECK(logLik == doctest::Approx(-20635.0));
            }

            THEN("g has expected dimensions")
            {
                REQUIRE(g.rows() == nx);
                REQUIRE(g.cols() == 1);

                AND_THEN("g has expected values")
                {
                    Eigen::VectorXd g_exp(nx);
                    g_exp << -3.54707518022088, 0.0, 0.0;
                    CAPTURE_EIGEN(g);
                    CAPTURE_EIGEN(g_exp);
                    CHECK(g.isApprox(g_exp));
                }
            }

            THEN("H has expected dimension")
            {
                REQUIRE(H.rows() == nx);
                REQUIRE(H.cols() == nx);

                AND_THEN("H has expected values")
                {
                    Eigen::MatrixXd H_exp(nx, nx);
                    H_exp << -0.000399073115164966, 0.0, 0.0,
                                               0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0;
                    CAPTURE_EIGEN(H);
                    CAPTURE_EIGEN(H_exp);
                    CHECK(H.isApprox(H_exp));
                }
            }
        }
    }
}
