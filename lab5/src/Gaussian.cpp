// Tip: Only include headers needed to parse this implementation only
#include <cassert>
#include <Eigen/Core>
#include <Eigen/QR>

#include "Gaussian.h"

Gaussian::Gaussian()
{

}

Gaussian::Gaussian(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S)
    : mu_(mu)
    , S_(S)
{
    assert(mu_.size() == S_.cols());
    assert(S_.isUpperTriangular());
}

Eigen::VectorXd Gaussian::mean() const
{
    return mu_;
}

Eigen::MatrixXd Gaussian::sqrtCov() const
{
    return S_;
}

Eigen::MatrixXd Gaussian::cov() const
{
    return S_.transpose()*S_;
}

Gaussian Gaussian::add(const Gaussian & other) const
{
    // Gaussian out;
    // return out;

    // i) check Ss have same number of cols
    const int n = static_cast<int>(mu_.size()); //length of mean vector
    assert(other.mu_.size() == n); // same mean dimensions
    assert(S_.cols() == n && other.S_.cols() == n); // same sqrtCov S have same cols

    // stack 
    const int m1 = static_cast<int>(S_.rows()); // number rows in S1
    const int m2 = static_cast<int>(other.S_.rows()); // number rows in S2
    Eigen::MatrixXd A(m1 + m2, n); // A = [S1, S2]^T
    if (m1) A.topRows(m1) = S_; // put S1 into A
    if (m2) A.bottomRows(m2) = other.S_; //put S2 into A

    // QR → R and use upper
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A); // QR factorisation of A
    Eigen::MatrixXd R = qr.matrixQR().template triangularView<Eigen::Upper>(); // extract upper tri part R from QR

    // get S, take top-n rows, extract R1 the top nxn block
    Eigen::MatrixXd S = R.topRows(n).template triangularView<Eigen::Upper>();

    // Add means
    Eigen::VectorXd mu = mu_ + other.mu_;
    return Gaussian(mu, S);
}

Gaussian Gaussian::marginalHead(int na) const
{
    //check na is in range, expects na + nb = n
    const int n = mu_.size();
    assert(na >= 0 && na <= n && "marginalHead: na out of range");

    //mua is first na elements of mu
    Eigen::VectorXd mu_a = mu_.head(na);
    // Sa is the top-left na x na block of S
    Eigen::MatrixXd S_a  = sqrtCov().topLeftCorner(na, na);

    // return N_{1/2}(x_a; μ_a, S_a)
    return Gaussian(mu_a, S_a);
}

Gaussian Gaussian::conditionalTailGivenHead(const Eigen::VectorXd & xa) const
{
    // Ensure input dimension is consistent with partition
    const int n = mu_.size();                  // total dimension
    const int na = xa.size();                 // dimension of head
    const int nb = n - na;                    // dimension of tail
    assert(na > 0 && nb > 0);                 // must be a valid

    // Partition mean into head (mu_a) and tail (mu_b)
    Eigen::VectorXd mua = mu_.head(na);        // first na entries
    Eigen::VectorXd mub = mu_.tail(nb);        // last nb entries

    // Partition S (upper-triangular square-root covariance) into blocks:
    // [ S1  S2 ]
    // [  0  S3 ]
    Eigen::MatrixXd S1 = S_.topLeftCorner(na, na);
    Eigen::MatrixXd S2 = S_.topRightCorner(na, nb);
    Eigen::MatrixXd S3 = S_.bottomRightCorner(nb, nb);

    // --- Conditional mean (μ_b|a) from (14) ---
    //
    // μ_b|a = μ_b + S₂ᵀ * S₁⁻ᵀ * (x_a – μ_a)
    //
    // Use triangular solve: S1.triangularView<Upper>().transpose().solve(...)
    Eigen::VectorXd delta = xa - mua;
    Eigen::VectorXd temp = S1.triangularView<Eigen::Upper>().transpose().solve(delta);
    Eigen::VectorXd mub_a = mub + S2.transpose() * temp;

    // --- Conditional square-root covariance (S_b|a) from (14) ---
    //
    // S_b|a = S₃  (lower-right block of S)
    return Gaussian(mub_a, S3);    
}

Gaussian Gaussian::permute(const Eigen::ArrayXi & idx) const
{
    // n  = original state size
    // nI = number of indices selected (can be a full permutation or a subset)
    const int n  = static_cast<int>(mu_.size());
    const int nI = static_cast<int>(idx.size());
    assert(nI > 0);

    // Bounds check
    for (int j = 0; j < nI; ++j) {
        assert(idx[j] >= 0 && idx[j] < n && "permute: idx out of range");
    }

    //Mean: gather mu_I = mu(idx)
    Eigen::VectorXd muI(nI);
    for (int j = 0; j < nI; ++j) {
        muI[j] = mu_[idx[j]];
    }

    // Square-root gather: S_:,I = S_[:, idx]
    Eigen::MatrixXd S_col(S_.rows(), nI);
    for (int j = 0; j < nI; ++j) {
        S_col.col(j) = S_.col(idx[j]);
    }

    // Q-less QR, R upper triangular
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(S_col);
    const Eigen::MatrixXd Rupper = qr.matrixQR().template triangularView<Eigen::Upper>();

    // square nI×nI top block
    const Eigen::MatrixXd SI = Rupper.topRows(nI).template triangularView<Eigen::Upper>();

    return Gaussian(muI, SI);
}

