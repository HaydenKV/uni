#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Cholesky>
#include "../../src/funcmin.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

TEST_CASE("TRSSQRTSPARSE: Return Newton step when it is inside trust region")
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
    Eigen::PermutationMatrix<Eigen::Dynamic> Pi(4);
    Pi.indices() << 2, 3, 0, 1;
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> rtr(Pi*H*Pi.transpose());
    Eigen::SparseMatrix<double> Xi = Eigen::MatrixXd(rtr.matrixU()).sparseView();
    
    double Delta = 2;
    Eigen::VectorXd pNewton = -H.llt().solve(g);
    
    REQUIRE((Xi*Pi.transpose()*pNewton).norm() <= Delta);

    Eigen::VectorXd p(4);
    int retval = funcmin::trsSqrtSparse(Xi, Pi, g, Delta, p);

    CHECK(retval == 0);

    CAPTURE_EIGEN(p);
    CAPTURE_EIGEN(pNewton);
    CHECK(p.isApprox(pNewton));
}

TEST_CASE("TRSSQRTSPARSE: Step length equals trust region radius when Newton step outside trust region")
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
    Eigen::PermutationMatrix<Eigen::Dynamic> Pi(4);
    Pi.indices() << 2, 3, 0, 1;
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> rtr(Pi*H*Pi.transpose());
    Eigen::SparseMatrix<double> Xi = Eigen::MatrixXd(rtr.matrixU()).sparseView();

    Eigen::VectorXd pNewton = -H.llt().solve(g);
    double Delta = 0.9*pNewton.norm();

    REQUIRE((Xi*Pi.transpose()*pNewton).norm() > Delta);

    Eigen::VectorXd p(4);
    int retval = funcmin::trsSqrtSparse(Xi, Pi, g, Delta, p);

    CHECK(retval == 0);

    // Want ||Xi*Pi^T*p|| == Delta
    CHECK((Xi*Pi.transpose()*p).norm() == doctest::Approx(Delta));
}
