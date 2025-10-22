#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "../../src/funcmin.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

TEST_CASE("TRSEIG: Return Newton step when it is inside trust region")
{
    Eigen::VectorXd g(4);
    g << 1,
         1,
         1,
         1;

    Eigen::MatrixXd H(4, 4);
    H << 1, 0, 0, 0,
         0, 2, 0, 0,
         0, 0, 3, 0,
         0, 0, 0, 4;
    
    double Delta = 2;
    Eigen::VectorXd pNewton = -H.llt().solve(g);
    REQUIRE(pNewton.norm() <= Delta);
    
    Eigen::VectorXd p(4);
    int retval = funcmin::trsEig(H, g, Delta, p);

    CHECK(retval == 0);

    CAPTURE_EIGEN(p);
    CAPTURE_EIGEN(pNewton);
    CHECK(p.isApprox(pNewton));
}

TEST_CASE("TRSEIG: Step length equals trust region radius when Newton step outside trust region")
{
    Eigen::VectorXd g(4);
    g << 1,
         1,
         1,
         1;

    Eigen::MatrixXd H(4, 4);
    H << 1, 0, 0, 0,
         0, 2, 0, 0,
         0, 0, 3, 0,
         0, 0, 0, 4;

    Eigen::VectorXd pNewton = -H.llt().solve(g);
    double Delta = 0.9*pNewton.norm();
    REQUIRE(pNewton.norm() > Delta);

    Eigen::VectorXd p(4);
    int retval = funcmin::trsEig(H, g, Delta, p);

    CHECK(retval == 0);
    CHECK(p.norm() == doctest::Approx(Delta));
}

TEST_CASE("TRSEIG: Step length equals trust region radius when nonconvex")
{
    Eigen::VectorXd g(4);
    g << 1,
         1,
         1,
         1;

    Eigen::MatrixXd H(4,4);
    H << -2,  0,  0,  0,
          0, -1,  0,  0,
          0,  0,  0,  0,
          0,  0,  0,  1;

    double Delta = 2;

    Eigen::VectorXd p(4);
    int retval = funcmin::trsEig(H, g, Delta, p);

    CHECK(retval == 0);
    CHECK(p.norm() == doctest::Approx(Delta));
}

TEST_CASE("TRSEIG: Step length equals trust region radius when nonconvex (hard case)")
{
    Eigen::VectorXd g(4);
    g << 0,
         1,
         1,
         1;

    Eigen::MatrixXd H(4, 4);
    H << -2,  0,  0,  0,
          0, -1,  0,  0,
          0,  0,  0,  0,
          0,  0,  0,  1;

    double Delta = 2;

    Eigen::VectorXd p(4);
    int retval = funcmin::trsEig(H, g, Delta, p);

    CHECK(retval == 0);
    CHECK(p.norm() == doctest::Approx(Delta));
}
