#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "../../src/funcmin.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

TEST_CASE("TRSSQRTINV: Return Newton step when it is inside trust region")
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
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> rtr(H);
    Eigen::MatrixXd Xi = rtr.matrixU();
    Eigen::MatrixXd S = Xi.triangularView<Eigen::Upper>().transpose().solve(Eigen::MatrixXd::Identity(4, 4));
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(S);    // In-place QR decomposition
    S = S.triangularView<Eigen::Upper>();                       // Safe aliasing
    
    double Delta = 2;
    Eigen::VectorXd pNewton = -H.llt().solve(g);
    REQUIRE((Xi*pNewton).norm() <= Delta);
    
    Eigen::VectorXd p(4);
    int retval = funcmin::trsSqrtInv(S, g, Delta, p);

    CHECK(retval == 0);

    CAPTURE_EIGEN(p);
    CAPTURE_EIGEN(pNewton);
    CHECK(p.isApprox(pNewton));
}

TEST_CASE("TRSSQRTINV: Step length equals trust region radius when Newton step outside trust region")
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
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> rtr(H);
    Eigen::MatrixXd Xi = rtr.matrixU();
    Eigen::MatrixXd S = Xi.triangularView<Eigen::Upper>().transpose().solve(Eigen::MatrixXd::Identity(4, 4));
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(S);    // In-place QR decomposition
    S = S.triangularView<Eigen::Upper>();                       // Safe aliasing

    Eigen::VectorXd pNewton = -H.llt().solve(g);
    double Delta = 0.9*pNewton.norm();
    REQUIRE((Xi*pNewton).norm() > Delta);

    Eigen::VectorXd p(4);
    int retval = funcmin::trsSqrtInv(S, g, Delta, p);

    CHECK(retval == 0);

    // Want ||Xi*p|| == Delta
    CHECK((Xi*p).norm() == doctest::Approx(Delta));
}
