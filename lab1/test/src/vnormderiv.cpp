#include <doctest/doctest.h>
#include <string>
#include <Eigen/Core>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include "../../src/vnorm.h"

TEST_CASE("Vector norm gradient")
{
    for (const Eigen::Index n: {10U, 100U, 1000U})
    {
        Eigen::VectorXd x(n);
        int v = 0;
        for (Eigen::Index i = 0; i < x.size(); ++i)
            x(i) = ++v;
        
        VectorNorm vnfunc;
        VectorNormFwdAutoDiff vnfuncFAD;
        VectorNormRevAutoDiff vnfuncRAD;
        Eigen::VectorXd g, gFAD, gRAD;

        ankerl::nanobench::Bench bench;
        bench.title("Vector norm gradient [n=" + std::to_string(n) + "]");
        bench.performanceCounters(false);
        bench.relative(true);

        // Analyical derivatives
        bench.run("Analytical derivative", [&] {
            vnfunc(x, g);
        });

        // Forward-mode autodifferentiation
        bench.run("Forward-mode autodiff", [&] {
            vnfuncFAD(x, gFAD);
        });
        CHECK(gFAD.isApprox(g));

        // Reverse-mode autodifferentiation
        bench.run("Reverse-mode autodiff", [&] {
            vnfuncRAD(x, gRAD);
        });
        CHECK(gRAD.isApprox(g));
    }
}

TEST_CASE("Vector norm Hessian")
{
    for (const Eigen::Index n: {20U, 50U, 100U, 200U})
    {
        Eigen::VectorXd x(n);
        int v = 0;
        for (Eigen::Index i = 0; i < x.size(); ++i)
            x(i) = ++v;
        
        VectorNorm vnfunc;
        VectorNormFwdAutoDiff vnfuncFAD;
        VectorNormRevAutoDiff vnfuncRAD;
        Eigen::VectorXd g, gFAD, gRAD;
        Eigen::MatrixXd H, HFAD, HRAD;

        ankerl::nanobench::Bench bench;
        bench.title("Vector norm Hessian [n=" + std::to_string(n) + "]");
        bench.relative(true);

        // Analyical derivatives
        bench.run("Analytical derivative", [&] {
            vnfunc(x, g, H);
        });

        // Forward-mode autodifferentiation
        bench.run("Forward-mode autodiff", [&] {
            vnfuncFAD(x, gFAD, HFAD);
        });
        CHECK(gFAD.isApprox(g));
        CHECK(HFAD.isApprox(H));

        // Reverse-mode autodifferentiation
        bench.run("Reverse-mode autodiff", [&] {
            vnfuncRAD(x, gRAD, HRAD);
        });
        CHECK(gRAD.isApprox(g));
        CHECK(HRAD.isApprox(H));
    }
}