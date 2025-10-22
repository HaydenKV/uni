#include <cassert>
#include <cstdlib>
#include <print>
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]") // Format like MATLAB
#include <Eigen/Core>
#include "to_string.hpp"

int main(int argc, char *argv[])
{
    std::println("Eigen version: {}.{}.{}", 
                 EIGEN_WORLD_VERSION, 
                 EIGEN_MAJOR_VERSION, 
                 EIGEN_MINOR_VERSION);

    std::println("Create a column vector:");
    Eigen::VectorXd x(3); // allocate 3-long vector
    x << 1.0, 3.2, 0.01; // assign with <<
    std::println("x = \n{}\n", to_string(x));

    std::println("Create a matrix:");
    Eigen::MatrixXd A;
    Eigen::Vector4d i; i << 1, 2, 3, 4; // i = [1 2 3 4]^T // vector4d pre defined fixed size 4 column vec and assign with <<
    Eigen::RowVector3d j; j << 1, 2, 3; //j = [1 2 3] // pre defined 3 row vec
    A = i * j;  // 4x3 A(i,j)

    std::println("A.size() = {}", A.size());
    std::println("A.rows() = {}", A.rows());
    std::println("A.cols() = {}", A.cols());
    std::println("A = \n{}\n", to_string(A));
    std::println("A.transpose() = \n{}\n", to_string(A.transpose()));

    std::println("Matrix multiplication:");
    Eigen::VectorXd Ax = A * x;
    std::println("A*x = \n{}\n", to_string(Ax));

    std::println("Matrix concatenation:");
    Eigen::MatrixXd B(4, 6); // matrix of size X
    B << A, A; //assign
    std::println("B = \n{}\n", to_string(B));
    Eigen::MatrixXd C(8, 3);
    C << A, A;
    std::println("C = \n{}\n", to_string(C));

    std::println("Submatrix via block:");
    Eigen::MatrixXd D;
    Eigen::MatrixXd D_block = B.block(1, 2, 1, 3); // (start row 1, start col 2, n0 row 1, n0 col 3)
    std::println("D block = \n{}\n", to_string(D_block));
    std::println("Submatrix via slicing:");
    Eigen::MatrixXd D_slice = B(Eigen::seq(1, 1), Eigen::seq(2, 4)); // row 1, cols 2..4
    std::println("D slice = \n{}\n", to_string(D_slice));
    D = D_block;
    std::println("D using block = \n{}\n", to_string(D));


    std::println("Broadcasting:");
    Eigen::VectorXd v(6); // size X (6) column vec
    v << 1, 3, 5, 7, 4, 6;
    Eigen::MatrixXd E = B.rowwise() + v.transpose(); // broadcasts 1x6 (v) across all rows of B
    std::println("E = \n{}\n", to_string(E));

    std::println("Index subscripting:");
    Eigen::ArrayXi r(4); r << 0, 2, 1, 3; // convert to 0 based and -1
    Eigen::ArrayXi c(6); c << 0, 3, 1, 4, 2, 5;
    Eigen::MatrixXd F = B(r, c);
    std::println("F = \n{}\n", to_string(F));

    std::println("Memory mapping:");
    float array[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> G(array); // map memory into G, Rowmajor tells the map type
    array[2] = -3.0f;               // Change an element in the raw storage
    assert(array[2] == G(0,2));     // Ensure the change is reflected in the view
    G(2,0) = -7.0f;                 // Change an element via the view
    assert(G(2,0) == array[6]);     // Ensure the change is reflected in the raw storage
    std::println("G = \n{}\n", to_string(G));

    return EXIT_SUCCESS;
}
