#include <doctest/doctest.h>
#include <Eigen/Core>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include "../../src/rosenbrock.h"

SCENARIO("RosenbrockAnalytical at origin")
{
    Eigen::VectorXd x(2);

    GIVEN("x = (0, 0)")
    {
        x << 0, 0;

        RosenbrockAnalytical func;
        double f;

        // One argument
        WHEN("Evaluating f = RosenbrockAnalytical(x)")
        {
            f = func(x);
            THEN("f matches expected value")
            {
                CHECK(f == doctest::Approx(1.0));
            }
        }

        // Two arguments
        WHEN("Evaluating f = RosenbrockAnalytical(x, g)")
        {
            Eigen::VectorXd g;
            f = func(x, g);
            THEN("f matches expected value")
            {
                CHECK(f == doctest::Approx(1.0));
            }
            THEN("g matches expected values")
            {
                REQUIRE(g.rows() == 2);
                REQUIRE(g.cols() == 1);
                CHECK(g(0) == doctest::Approx(-2.0));
                CHECK(g(1) == doctest::Approx(0.0));
            }
        }
        
        // Three arguments
        WHEN("Evaluating f = RosenbrockAnalytical(x, g, H)")
        {
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            f = func(x, g, H);
            THEN("f matches expected value")
            {
                CHECK(f == doctest::Approx(1.0));
            }
            THEN("g matches expected values")
            {
                REQUIRE(g.rows() == 2);
                REQUIRE(g.cols() == 1);
                CHECK(g(0) == doctest::Approx(-2.0));
                CHECK(g(1) == doctest::Approx(0.0));
            }
            THEN("H matches expected values")
            {
                REQUIRE(H.rows() == 2);
                REQUIRE(H.cols() == 2);
                CHECK(H(0, 0) == doctest::Approx(2.0));
                CHECK(H(0, 1) == doctest::Approx(0.0));
                CHECK(H(1, 0) == doctest::Approx(0.0));
                CHECK(H(1, 1) == doctest::Approx(200));
            }
        }
    }
}

SCENARIO("RosenbrockAnalytical at minimiser")
{
    Eigen::VectorXd x(2);

    GIVEN("x = (1, 1)")
    {
        x << 1, 1;

        RosenbrockAnalytical func;
        double f;

        // One argument
        WHEN("Evaluating f = RosenbrockAnalytical(x)")
        {
            f = func(x);
            THEN("f matches expected value")
            {
                CHECK(f == doctest::Approx(0.0));
            }
        }

        // Two arguments
        WHEN("Evaluating f = RosenbrockAnalytical(x, g)")
        {
            Eigen::VectorXd g;
            f = func(x, g);
            THEN("f matches expected value")
            {
                CHECK(f == doctest::Approx(0.0));
            }
            THEN("g matches expected values")
            {
                REQUIRE(g.rows() == 2);
                REQUIRE(g.cols() == 1);
                CHECK(g(0) == doctest::Approx(0.0));
                CHECK(g(1) == doctest::Approx(0.0));
            }
        }
        
        // Three arguments
        WHEN("Evaluating f = RosenbrockAnalytical(x, g, H)")
        {
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            f = func(x, g, H);
            THEN("f matches expected value")
            {
                CHECK(f == doctest::Approx(0.0));
            }
            THEN("g matches expected values")
            {
                REQUIRE(g.rows() == 2);
                REQUIRE(g.cols() == 1);
                CHECK(g(0) == doctest::Approx(0.0));
                CHECK(g(1) == doctest::Approx(0.0));
            }
            THEN("H matches expected values")
            {
                REQUIRE(H.rows() == 2);
                REQUIRE(H.cols() == 2);
                CHECK(H(0, 0) == doctest::Approx(802.0));
                CHECK(H(0, 1) == doctest::Approx(-400.0));
                CHECK(H(1, 0) == doctest::Approx(-400.0));
                CHECK(H(1, 1) == doctest::Approx(200));
            }
        }
    }
}

TEST_CASE("Rosenbrock first derivative performance")
{
    Eigen::VectorXd x(2);
    x << 0, 0;

    ankerl::nanobench::Bench bench;
    bench.title("Rosenbrock gradient only");
    bench.relative(true);
    bench.performanceCounters(false);

    RosenbrockAnalytical func;
    double f;
    Eigen::VectorXd g;
    bench.run("Analytical derivatives", [&] {
        f = func(x, g);
    });

    RosenbrockFwdAutoDiff funcFwd;
    double fFwd;
    Eigen::VectorXd gFwd;
    bench.run("Forward-mode autodiff", [&] {
        fFwd = funcFwd(x, gFwd);
    });
    CHECK(fFwd == doctest::Approx(f));
    CHECK(gFwd.isApprox(g));

    RosenbrockRevAutoDiff funcRev;
    double fRev;
    Eigen::VectorXd gRev;
    bench.run("Reverse-mode autodiff", [&] {
        fRev = funcRev(x, gRev);
    });
    CHECK(fRev == doctest::Approx(f));
    CHECK(gRev.isApprox(g));
}

TEST_CASE("Rosenbrock second derivative performance")
{
    Eigen::VectorXd x(2);
    x << 0, 0;

    ankerl::nanobench::Bench bench;
    bench.title("Rosenbrock gradient and Hessian");
    bench.relative(true);

    RosenbrockAnalytical func;
    double f;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    bench.run("Analytical derivatives", [&] {
        f = func(x, g, H);
    });

    RosenbrockFwdAutoDiff funcFwd;
    double fFwd;
    Eigen::VectorXd gFwd;
    Eigen::MatrixXd HFwd;
    bench.run("Forward-mode autodiff", [&] {
        fFwd = funcFwd(x, gFwd, HFwd);
    });
    CHECK(fFwd == doctest::Approx(f));
    CHECK(gFwd.isApprox(g));
    CHECK(HFwd.isApprox(H));

    RosenbrockRevAutoDiff funcRev;
    double fRev;
    Eigen::VectorXd gRev;
    Eigen::MatrixXd HRev;
    bench.run("Reverse-mode autodiff", [&] {
        fRev = funcRev(x, gRev, HRev);
    });
    CHECK(fRev == doctest::Approx(f));
    CHECK(gRev.isApprox(g));
    CHECK(HRev.isApprox(H));
}
