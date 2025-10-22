#include <doctest/doctest.h>
#include <cstddef>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemBallistic.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("SystemBallistic")
{
    const std::size_t n = 3;
    GaussianInfo<double> p0 = GaussianInfo<double>::fromSqrtInfo(Eigen::MatrixXd::Zero(n, n));
    SystemBallistic system(p0);

    GIVEN("Time, state and input")
    {
        double t = 0.0;
        Eigen::VectorXd x(n);
        x <<  14000,
               -450,
             0.0005;
        Eigen::VectorXd u(0);

        WHEN("Evaluating f = system.dynamics(t, x, u, J)")
        {
            Eigen::MatrixXd J;
            Eigen::VectorXd f = system.dynamics(t, x, u, J);

            THEN("f has expected dimensions")
            {
                REQUIRE(f.rows() == n);
                REQUIRE(f.cols() == 1);

                AND_THEN("f has expected values")
                {
                    Eigen::VectorXd f_expected(n);
                    f_expected <<             -450,
                                  2.51399051183737,
                                                 0;
                    CAPTURE_EIGEN(f);
                    CAPTURE_EIGEN(f_expected);
                    CHECK(f.isApprox(f_expected));
                }
            }

            THEN("J has expected dimensions")
            {
                REQUIRE(J.rows() == n);
                REQUIRE(J.cols() == n);

                AND_THEN("J has expected values")
                {
                    Eigen::MatrixXd J_expected(n, n);
                    J_expected <<                    0,                    1,                    0,
                                  -0.00172993748945584,  -0.0547732911637217,     24647.9810236747,
                                                     0,                    0,                    0;
                    CAPTURE_EIGEN(J);
                    CAPTURE_EIGEN(J_expected);
                    CHECK(J.isApprox(J_expected));
                }
            }
        }
    }
}
