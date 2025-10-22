/**
 * @file Gaussian.hpp
 * @brief Defines the Gaussian class for representing and manipulating Gaussian distributions.
 *
 * This file contains the implementation of the Gaussian class, which represents
 * a multivariate Gaussian (normal) distribution. The class provides various
 * operations such as marginalization, conditioning, and transformation of
 * Gaussian distributions. It also includes methods for computing log-likelihoods,
 * gradients, and Hessians, as well as utilities for working with confidence regions.
 *
 * The Gaussian class is templated on the scalar type, allowing for flexibility
 * in the numeric precision used for calculations and to enable autodiff support.
 */

#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include <cstddef>
#include <cmath>
#include <ctime>
#include <numbers>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Cholesky>
#include <Eigen/LU> // for .determinant() and .inverse(), which you shouldn't need to use
#include "GaussianBase.hpp"

/**
 * @brief Represents a Gaussian distribution.
 * 
 * This class implements a Gaussian (normal) distribution with support for
 * various operations such as marginalization, conditioning, and transformation.
 * It inherits from GaussianBase and provides additional functionality specific
 * to Gaussian distributions.
 * 
 * @tparam Scalar The scalar type used for calculations, default is double.
 */
template <typename Scalar = double>
class Gaussian : public GaussianBase<Scalar>
{
public:
    virtual ~Gaussian() override = default;

    using GaussianBase<Scalar>::normcdf;
    using GaussianBase<Scalar>::chi2inv;
protected:
    /**
     * @brief Default constructor for Gaussian distribution.
     * 
     * Constructs a Gaussian distribution without initializing mean or covariance.
     */
    Gaussian()
        : GaussianBase<Scalar>()
    {}

    /**
     * @brief Constructor for Gaussian distribution with specified dimension.
     * 
     * @param n The dimension of the Gaussian distribution.
     * 
     * Constructs a Gaussian distribution with zero mean and uninitialized covariance
     * of the specified dimension.
     */
    explicit Gaussian(std::size_t n)
        : GaussianBase<Scalar>()
        , mu_(n)
        , S_(n, n)
    {}

    /**
     * @brief Construct a Gaussian distribution with zero mean and given square root covariance matrix.
     * 
     * @param S The square root covariance matrix (upper triangular).
     */
    explicit Gaussian(const Eigen::MatrixX<Scalar> & S)
        : GaussianBase<Scalar>()
        , mu_(Eigen::VectorX<Scalar>::Zero(S.cols()))
        , S_(S)
    {
        assert(mu_.size() == S_.cols());
        assert(S_.isUpperTriangular());
    }

    /**
     * @brief Construct a Gaussian distribution with given mean and square root covariance matrix.
     * 
     * @param mu The mean vector of the Gaussian distribution.
     * @param S The square root covariance matrix (upper triangular).
     * 
     * This constructor initializes a Gaussian distribution with the specified mean vector
     * and square root covariance matrix. It ensures that the dimensions of the input
     * parameters are consistent.
     */
    Gaussian(const Eigen::VectorX<Scalar> & mu, const Eigen::MatrixX<Scalar> & S)
        : GaussianBase<Scalar>()
        , mu_(mu)
        , S_(S)
    {
        assert(mu_.size() == S_.cols());
        assert(S_.isUpperTriangular());
    }
    /**
     * @brief Declare Gaussian class as a friend for all scalar types.
     * 
     * This allows the Gaussian class with different scalar types to access
     * private members of each other, facilitating type conversion operations.
     */
    template <typename OtherScalar> friend class Gaussian;

    /**
     * @brief Convert a Gaussian distribution from one scalar type to another.
     * 
     * @tparam OtherScalar The scalar type of the source Gaussian distribution.
     * @param p The source Gaussian distribution to convert from.
     * 
     * This constructor creates a new Gaussian distribution by converting
     * the mean vector and square root covariance matrix from the source
     * scalar type to the current scalar type.
     * 
     * @note This constructor performs type checking to ensure the dimensions
     * of the converted distribution are consistent.
     */
    template <typename OtherScalar>
    explicit Gaussian(const Gaussian<OtherScalar> & p)
        : GaussianBase<Scalar>()
        , mu_(p.mu_.template cast<Scalar>())
        , S_(p.S_.template cast<Scalar>())
    {
        assert(mu_.size() == S_.cols());
        assert(S_.isUpperTriangular());
    }
public:
    /**
     * @brief Casts the Gaussian distribution to a different scalar type.
     * 
     * This method creates a new Gaussian distribution with a different scalar type,
     * while maintaining the same distribution parameters.
     * 
     * @tparam OtherScalar The target scalar type for the cast operation.
     * @return A new Gaussian object with the specified scalar type.
     */
    template <typename OtherScalar>
    Gaussian<OtherScalar> cast() const
    {
        return Gaussian<OtherScalar>(*this);
    }

    //
    // Two-argument factories
    //
    
    /**
     * @brief Creates a Gaussian object from square root moment parameters.
     * 
     * This static factory method constructs a Gaussian object using the mean vector
     * and the square root of the covariance matrix (upper triangular).
     * 
     * @param mu The mean vector of the Gaussian distribution.
     * @param S The square root of the covariance matrix (upper triangular).
     * @return A new Gaussian object representing the Gaussian distribution.
     * 
     * @note This method performs dimension checks to ensure consistency between
     *       the input parameters.
     */
    static Gaussian fromSqrtMoment(const Eigen::VectorX<Scalar> & mu, const Eigen::MatrixX<Scalar> & S)
    {
        assert(mu.size() == S.cols());
        assert(S.isUpperTriangular());
        Gaussian out;
        out.mu_ = mu;
        out.S_ = S;
        return out;
    }

    /**
     * @brief Creates a Gaussian object from moment parameters.
     * 
     * This static factory method constructs a Gaussian object using the mean vector
     * and the covariance matrix.
     * 
     * @param mu The mean vector of the Gaussian distribution.
     * @param P The covariance matrix of the Gaussian distribution.
     * @return A new Gaussian object representing the Gaussian distribution.
     * 
     * @note This method performs dimension checks to ensure consistency between
     *       the input parameters and computes the square root of the covariance matrix
     *       using Cholesky decomposition.
     */
    static Gaussian fromMoment(const Eigen::VectorX<Scalar> & mu, const Eigen::MatrixX<Scalar> & P)
    {
        assert(mu.size() == P.cols());
        assert(P.rows() == P.cols());
        Gaussian out;
        out.mu_ = mu;
        // Let S be an upper-triangular matrix such that S^T*S = P
        Eigen::LLT<Eigen::MatrixX<Scalar>, Eigen::Upper> llt(P);
        out.S_ = llt.matrixU();
        return out;
    }

    /**
     * @brief Creates a Gaussian object from square root information parameters.
     * 
     * This static factory method constructs a Gaussian object using the square root
     * information vector and the square root information matrix.
     * 
     * @param nu The square root information vector.
     * @param Xi The square root information matrix (upper triangular).
     * @return A new Gaussian object representing the Gaussian distribution.
     * 
     * @note This method performs dimension checks to ensure consistency between
     *       the input parameters and computes the mean and square root covariance
     *       matrix from the square root information parameters.
     */
    static Gaussian fromSqrtInfo(const Eigen::VectorX<Scalar> & nu, const Eigen::MatrixX<Scalar> & Xi)
    {
        assert(nu.size() == Xi.cols());
        assert(Xi.isUpperTriangular());
        Gaussian out;
        // Solve Xi*mu = nu
        out.mu_ = Xi.template triangularView<Eigen::Upper>().solve(nu);
        // S = qr(Xi^{-T})
        out.S_ = Xi.template triangularView<Eigen::Upper>().transpose().solve(
            Eigen::MatrixX<Scalar>::Identity(Xi.cols(), Xi.cols())
        );
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(out.S_);        // In-place QR decomposition
        out.S_ = out.S_.template triangularView<Eigen::Upper>();                    // Safe aliasing
        return out;
    }

    /**
     * @brief Creates a Gaussian object from information parameters.
     * 
     * This static factory method constructs a Gaussian object using the information vector
     * and the information matrix.
     * 
     * @param eta The information vector of the Gaussian distribution.
     * @param Lambda The information matrix of the Gaussian distribution.
     * @return A new Gaussian object representing the Gaussian distribution.
     * 
     * @note This method performs dimension checks to ensure consistency between
     *       the input parameters and computes the square root information matrix
     *       using Cholesky decomposition before calling fromSqrtInfo.
     */
    static Gaussian fromInfo(const Eigen::VectorX<Scalar> & eta, const Eigen::MatrixX<Scalar> & Lambda)
    {
        assert(eta.size() == Lambda.cols());
        assert(Lambda.rows() == Lambda.cols());
        // Let Xi be an upper-triangular matrix such that Xi^T*Xi = Lambda
        Eigen::LLT<Eigen::MatrixX<Scalar>, Eigen::Upper> llt(Lambda);
        Eigen::MatrixX<Scalar> Xi = llt.matrixU();
        // Solve Xi^T*nu = eta
        Eigen::VectorX<Scalar> nu = Xi.template triangularView<Eigen::Upper>().transpose().solve(eta);
        return fromSqrtInfo(nu, Xi);
    }

    //
    // One-argument factories
    //
    
    /**
     * @brief Creates a Gaussian object from square root covariance matrix with zero mean.
     * 
     * This static factory method constructs a Gaussian object using only the square root
     * of the covariance matrix (upper triangular) and assumes a zero mean vector.
     * 
     * @param S The square root of the covariance matrix (upper triangular).
     * @return A new Gaussian object representing the Gaussian distribution with zero mean.
     * 
     * @note This method internally calls fromSqrtMoment with a zero mean vector of 
     *       appropriate size.
     */
    static Gaussian fromSqrtMoment(const Eigen::MatrixX<Scalar> & S)
    {
        return fromSqrtMoment(Eigen::VectorX<Scalar>::Zero(S.cols()), S);
    }

    /**
     * @brief Creates a Gaussian object from covariance matrix with zero mean.
     * 
     * This static factory method constructs a Gaussian object using only the
     * covariance matrix and assumes a zero mean vector.
     * 
     * @param P The covariance matrix of the Gaussian distribution.
     * @return A new Gaussian object representing the Gaussian distribution with zero mean.
     * 
     * @note This method internally calls fromMoment with a zero mean vector of 
     *       appropriate size and the provided covariance matrix.
     */
    static Gaussian fromMoment(const Eigen::MatrixX<Scalar> & P)
    {
        return fromMoment(Eigen::VectorX<Scalar>::Zero(P.cols()), P);
    }

    /**
     * @brief Creates a Gaussian object from square root information matrix with zero information vector.
     * 
     * This static factory method constructs a Gaussian object using only the square root
     * of the information matrix (upper triangular) and assumes a zero information vector.
     * 
     * @param Xi The square root of the information matrix (upper triangular).
     * @return A new Gaussian object representing the Gaussian distribution with zero information vector.
     * 
     * @note This method internally calls fromSqrtInfo with a zero information vector of 
     *       appropriate size and the provided square root information matrix.
     */
    static Gaussian fromSqrtInfo(const Eigen::MatrixX<Scalar> & Xi)
    {
        return fromSqrtInfo(Eigen::VectorX<Scalar>::Zero(Xi.cols()), Xi);
    }

    /**
     * @brief Creates a Gaussian object from information matrix with zero information vector.
     * 
     * This static factory method constructs a Gaussian object using only the
     * information matrix and assumes a zero information vector.
     * 
     * @param Lambda The information matrix of the Gaussian distribution.
     * @return A new Gaussian object representing the Gaussian distribution with zero information vector.
     * 
     * @note This method internally calls fromInfo with a zero information vector of 
     *       appropriate size and the provided information matrix.
     */
    static Gaussian fromInfo(const Eigen::MatrixX<Scalar> & Lambda)
    {
        return fromInfo(Eigen::VectorX<Scalar>::Zero(Lambda.cols()), Lambda);
    }

    /**
     * @brief Get the dimension of the Gaussian distribution.
     * 
     * @return The dimension of the Gaussian distribution.
     */
    virtual Eigen::Index dim() const override
    {
        return S_.cols();
    }

    /**
     * @brief Get the mean vector of the Gaussian distribution.
     * 
     * @return The mean vector of the Gaussian distribution.
     */
    virtual Eigen::VectorX<Scalar> mean() const override
    {
        return mu_;
    }

    /**
     * @brief Get the square root of the covariance matrix.
     * 
     * @return The upper triangular square root of the covariance matrix.
     * 
     * The returned matrix \f$\mathbf{S}\f$ is upper triangular and satisfies \f$\mathbf{S}^\mathsf{T} \mathbf{S} = \mathbf{P}\f$,
     * where \f$\mathbf{P}\f$ is the covariance matrix.
     */
    virtual Eigen::MatrixX<Scalar> sqrtCov() const override
    {
        return S_;
    }

    /**
     * @brief Get the covariance matrix of the Gaussian distribution.
     * 
     * @return The covariance matrix of the Gaussian distribution.
     */
    virtual Eigen::MatrixX<Scalar> cov() const override
    {
        return S_.transpose()*S_;
    }

    /**
     * @brief Get the information matrix of the Gaussian distribution.
     * 
     * @return The information matrix of the Gaussian distribution.
     */
    virtual Eigen::MatrixX<Scalar> infoMat() const override
    {
        // Lambda = (S^T*S)^{-1} = S^{-1}*S^{-T}
        return S_.template triangularView<Eigen::Upper>().solve(
            S_.template triangularView<Eigen::Upper>().transpose().solve(
                Eigen::MatrixX<Scalar>::Identity(S_.cols(), S_.cols())
            )
        );
    }

    /**
     * @brief Get the information vector of the Gaussian distribution.
     * 
     * @return The information vector of the Gaussian distribution.
     */
    virtual Eigen::VectorX<Scalar> infoVec() const override
    {
        // eta = (S^T*S)^{-1}*mu = S^{-1}*S^{-T}*mu
        return S_.template triangularView<Eigen::Upper>().solve(
            S_.template triangularView<Eigen::Upper>().transpose().solve(mu_)
        );
    }

    /**
     * @brief Get the square root of the information matrix.
     * 
     * @return The upper triangular square root of the information matrix.
     * 
     * The returned matrix \f$\boldsymbol\Xi\f$ is upper triangular and satisfies \f$\boldsymbol\Xi^\mathsf{T} \boldsymbol\Xi = \boldsymbol\Lambda\f$,
     * where \f$\boldsymbol\Lambda\f$ is the information matrix.
     */
    virtual Eigen::MatrixX<Scalar> sqrtInfoMat() const override
    {
        // qr(S^{-T})
        Eigen::MatrixX<Scalar> Xi = S_.template triangularView<Eigen::Upper>().transpose().solve(
            Eigen::MatrixX<Scalar>::Identity(S_.cols(), S_.cols())
        );
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(Xi);        // In-place QR decomposition
        Xi = Xi.template triangularView<Eigen::Upper>();                        // Safe aliasing
        return Xi;
    }

    /**
     * @brief Get the square root of the information vector.
     * 
     * @return The square root of the information vector of the Gaussian distribution.
     * 
     * The returned vector \f$\boldsymbol\nu\f$ satisfies \f$\boldsymbol\nu = \boldsymbol\Xi \boldsymbol\mu\f$,
     * where \f$\boldsymbol\Xi\f$ is the square root of the information matrix and \f$\boldsymbol\mu\f$ is the mean vector.
     */
    virtual Eigen::VectorX<Scalar> sqrtInfoVec() const override
    {
        // nu = Xi*mu
        return sqrtInfoMat()*mu_;
    }

    /**
     * @brief Creates a Gaussian object from a set of samples.
     * 
     * This static factory method constructs a Gaussian object by estimating
     * the mean and covariance from the provided sample data.
     * 
     * @param X The matrix of samples, where each column represents a sample
     *          and each row represents a dimension of the distribution.
     * @return A new Gaussian object representing the estimated distribution.
     * 
     * @note This method assumes that the samples are independent and
     *       identically distributed (i.i.d.) draws from a Gaussian distribution.
     */
    static Gaussian fromSamples(const Eigen::MatrixX<Scalar> & X)
    {
        const Eigen::Index n = X.rows();
        const Eigen::Index m = X.cols();

        Gaussian out;

        // Compute the sample mean
        out.mu_ = X.rowwise().mean();

        // Compute the sample square-root covariance
        Eigen::MatrixX<Scalar> SS = std::sqrt(1.0 / (m - 1))*(X.colwise() - out.mu_).transpose();
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(SS);   // In-place QR decomposition
        out.S_ = SS.topRows(n).template triangularView<Eigen::Upper>();
        return out;
    }

    /**
     * @brief Compute the marginal distribution of a subset of variables.
     *
     * Given the joint density p(x), this method returns the marginal density p(x(idx)).
     *
     * @tparam IndexType The type of the index container (e.g., std::vector<int>, Eigen::VectorXi, Eigen::ArrayXi)
     * @param idx The indices of the variables to marginalize over
     * @return The marginal Gaussian distribution of the selected variables
     *
     * @note The resulting distribution will have a dimension equal to the size of idx.
     */
    template <typename IndexType>
    Gaussian marginal(const IndexType & idx) const
    {
        Gaussian out;
        // TODO
        // out.mu_ = ???
        // out.S_ = ???

        // 1) Mean: μ_I (in the order of idx)
        out.mu_ = mu_(idx);

        // 2) Square-root covariance:
        //    Take the columns of S corresponding to idx → S(:,I) (n x n_I)
        const std::size_t nI = idx.size();
        Eigen::MatrixX<Scalar> S_colI = S_(Eigen::all, idx);

        // Q-less QR factorization: S_colI = Q * R (we only keep R, in-place)
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(S_colI);

        // Extract the upper-triangular part and take the top n_I rows as R1
        S_colI = S_colI.template triangularView<Eigen::Upper>();
        out.S_ = S_colI.topRows(static_cast<Eigen::Index>(nI)).template triangularView<Eigen::Upper>();
        return out;
    }

    /**
     * @brief Compute the conditional distribution given a subset of variables.
     *
     * Given the joint density p(x), this method returns the conditional density p(x(idxA) | x(idxB) = xB).
     *
     * @tparam IndexTypeA The type of the index container for set A (e.g., std::vector<int>, Eigen::VectorXi, Eigen::ArrayXi)
     * @tparam IndexTypeB The type of the index container for set B (e.g., std::vector<int>, Eigen::VectorXi, Eigen::ArrayXi)
     * @param idxA The indices of the variables to condition on (set A)
     * @param idxB The indices of the conditioning variables (set B)
     * @param xB The values of the conditioning variables
     * @return The conditional Gaussian distribution of x(idxA) given x(idxB) = xB
     *
     * @note The resulting distribution will have a dimension equal to the size of idxA.
     */
    template <typename IndexTypeA, typename IndexTypeB>
    Gaussian conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorX<Scalar> & xB) const
    {
        // // FIXME: The following implementation is in error, but it does pass some of the unit tests
        // Gaussian out;
        // out.mu_ = mu_(idxA) +
        //     S_(idxB, idxA).transpose()*
        //     S_(idxB, idxB).eval().template triangularView<Eigen::Upper>().transpose().solve(xB - mu_(idxB));
        // out.S_ = S_(idxA, idxA);
        // return out;

        Gaussian out;

        // Sizes and basic checks
        const Eigen::Index nA = static_cast<Eigen::Index>(idxA.size());
        const Eigen::Index nB = static_cast<Eigen::Index>(idxB.size());
        const Eigen::Index n  = dim();
        assert(nA + nB == n);
        assert(xB.size() == nB);

        // (1) Build the column concatenation [ S(:,B)  S(:,A) ] ("swap" A/B block order)
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> SS(n, nB + nA);
        SS.leftCols(nB)  = S_(Eigen::all, idxB);
        SS.rightCols(nA) = S_(Eigen::all, idxA);

        // (2) Q-less QR on [ S_B  S_A ]
        Eigen::HouseholderQR<Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>> qr(SS);

        // Extract the square R = [ S1 S2 ; 0 S3 ] from the top rows and keep only the upper triangle.
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> R =
            SS.topRows(nB + nA).template triangularView<Eigen::Upper>();

        // (3) Partition R into its blocks:
        //     R = [ R1  R2 ;
        //           0   R3 ]
        const auto R1 = R.topLeftCorner(nB, nB);        // upper-triangular
        const auto R2 = R.topRightCorner(nB, nA);   //(nB×nA)
        const auto R3 = R.bottomRightCorner(nA, nA);    // nA×nA, upper triangular

        // 4) Conditional mean: μ_A|B = μ_A + (R1^{-1} R2)^T (xB - μ_B)
        //    Compute K = R1 \ R2 via triangular solve, then K^T * deltaB (lab formula)
        // deltaB = xB - muB
        const Eigen::VectorX<Scalar> deltaB = xB - mu_(idxB);

        // Solve R1^T y = deltaB  (since R1 is upper-triangular, R1^T is lower-triangular)
        const Eigen::VectorX<Scalar> y =
            R1.template triangularView<Eigen::Upper>()
            .transpose()
            .solve(deltaB);

        // μ_A|B = μ_A + R2^T y   (mat-vec, no explicit K)
        out.mu_ = mu_(idxA) + R2.transpose() * y;


        // 5) Conditional sqrt covariance is R3 directly S_{A|B} = S3
        out.S_ = R3;

        return out;
    }

    /**
     * @brief Compute the conditional distribution given a subset of variables and their conditional distribution.
     *
     * Given the joint density p(x) and the conditional density p(x(idxB) | y) for some data y,
     * this method returns the conditional density p(x(idxA) | y).
     *
     * @tparam IndexTypeA The type of the index container for set A (e.g., std::vector<int>, Eigen::VectorXi, Eigen::ArrayXi)
     * @tparam IndexTypeB The type of the index container for set B (e.g., std::vector<int>, Eigen::VectorXi, Eigen::ArrayXi)
     * @param idxA The indices of the variables to condition on (set A)
     * @param idxB The indices of the variables with known conditional distribution (set B)
     * @param pxB_y The conditional Gaussian distribution of x(idxB) given y
     * @return The conditional Gaussian distribution of x(idxA) given y
     *
     * @note The resulting distribution will have a dimension equal to the size of idxA.
     */
    template <typename IndexTypeA, typename IndexTypeB>
    Gaussian conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Gaussian & pxB_y) const
    {
        const std::size_t & nA = idxA.size();
        const std::size_t & nB = idxB.size();

        // Form [S(:, idxB), S(:, idxA)]
        Eigen::MatrixX<Scalar> SS(dim(), nA + nB);
        SS << S_(Eigen::all, idxB), S_(Eigen::all, idxA);
        // Q-less QR yields
        // [S1, S2;
        //   0, S3]
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(SS);   // In-place QR decomposition

        // K = S1\S2
        Eigen::MatrixX<Scalar> K = SS.topLeftCorner(nB, nB).template triangularView<Eigen::Upper>().solve(SS.topRightCorner(nB, nA));
        Eigen::MatrixX<Scalar> S3 = SS.bottomRightCorner(nA, nA).template triangularView<Eigen::Upper>();

        // Form [pxB_y.S*K; S3]
        Eigen::MatrixX<Scalar> RR(nA + nB, nA);
        RR << pxB_y.S_ * K,
              S3;
        // Q-less QR yields
        // [R1;
        //   0]
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr_RR(RR);

        // p(x(idxA) | y) = N^-0.5(x(idxA); mu(idxA) + K.'*(pxB_y.mu - mu(idxB))), R1)
        Gaussian out;
        out.mu_ = mu_(idxA) + K.transpose()*(pxB_y.mean() - mu_(idxB));
        out.S_ = RR.topRows(nA).template triangularView<Eigen::Upper>();
        return out;

        // ALTERNATE APPROACH THAT MAY BE MORE ROBUST AROUND INF values
        // const Eigen::Index nA = static_cast<Eigen::Index>(idxA.size());
        // const Eigen::Index nB = static_cast<Eigen::Index>(idxB.size());

        // // (1) Build [ S(:,B)  S(:,A) ] and QR-partition it.
        // Eigen::MatrixX<Scalar> SS(dim(), nA + nB);
        // SS << S_(Eigen::all, idxB), S_(Eigen::all, idxA);

        // // Scale-in for robustness (prevents overflow in extreme tests).
        // Scalar s = SS.cwiseAbs().maxCoeff();
        // if (!(s > Scalar(0))) s = Scalar(1);
        // SS /= s;

        // // Q-less QR → R = [ S1 S2 ; 0 S3 ] stored in SS.
        // Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(SS);
        // SS = SS.template triangularView<Eigen::Upper>();

        // // Extract blocks (still scaled):
        // const Eigen::MatrixX<Scalar> S1  = SS.topLeftCorner(nB, nB)
        //                                     .template triangularView<Eigen::Upper>();
        // const Eigen::MatrixX<Scalar> S2  = SS.topRightCorner(nB, nA);
        // const Eigen::MatrixX<Scalar> S3s = SS.bottomRightCorner(nA, nA)
        //                                     .template triangularView<Eigen::Upper>();

        // // (2) K = S1^{-1} S2  via triangular solve (no explicit inverse).
        // Eigen::MatrixX<Scalar> K = S1.template triangularView<Eigen::Upper>().solve(S2);

        // // (3) Conditional mean: mu_A + K^T (mu_{B|y} - mu_B).
        // const Eigen::VectorX<Scalar> delta = pxB_y.mean() - mu_(idxB);
        // Eigen::VectorX<Scalar> muA_cond = mu_(idxA) + K.transpose() * delta;

        // // (4) Conditional sqrt-cov via a second QR on the stacked matrix:
        // //     RR = [ S_{B|y} * K ;  s * S3s ]  (we rescale S3s back here).
        // Eigen::MatrixX<Scalar> RR(nA + nB, nA);
        // RR << pxB_y.sqrtCov() * K,
        //     s * S3s;

        // Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr_RR(RR);
        // RR = RR.template triangularView<Eigen::Upper>();
        // Eigen::MatrixX<Scalar> R1 = RR.topRows(nA).template triangularView<Eigen::Upper>();

        // Gaussian out;
        // out.mu_ = muA_cond;
        // out.S_  = R1;
        // return out;
    }

    /**
     * @brief Propagate the Gaussian distribution through a nonlinear function.
     *
     * This method transforms the current Gaussian distribution p(x) through a given function y = h(x)
     * by propagating information through the affine transformation. It returns a new Gaussian distribution
     * representing p(y).
     *
     * @tparam Func The type of the function object.
     * @param h The function object representing the nonlinear transformation.
     *          It should take two arguments: the input vector and a reference to the Jacobian matrix.
     *          The function should return the transformed vector and populate the Jacobian matrix.
     * @return A new Gaussian distribution representing p(y).
     */
    template <typename Func>
    Gaussian affineTransform(Func h) const
    {
        Gaussian out;
        // Evaluate mean and Jacobian at μ: y = h(μ), C = ∂h/∂x |_{μ}
        Eigen::MatrixX<Scalar> C;
        out.mu_ = h(mu_, C);
        // TODO
        const Eigen::Index m = out.mu_.size();
        assert(C.rows() == m); //m = output dimension (size of y = h(μ))
        assert(C.cols() == dim()); //n = input/state dimension (size of x) dim() == n
        // Form S*C^T (n x m) (equ 8-9 lab doc)
        Eigen::MatrixX<Scalar> SCt = S_ * C.transpose();
        // Q-less Householder QR: SCt = Q * R, we keep R in-place
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(SCt);
        // Extract upper-triangular R and take its top m×m block as R1
        SCt = SCt.template triangularView<Eigen::Upper>();
        out.S_ = SCt.topRows(m).template triangularView<Eigen::Upper>();
        // out.S_ = C * S_ * C.transpose();
        // p(y) = N^{1/2}( y ; h(μ), S_y=R1 ) eq. (9)
        return out;
    }

    /**
     * @brief Compute the log-likelihood of a given point under the Gaussian distribution.
     *
     * This method calculates the natural logarithm of the probability density function
     * of the Gaussian distribution at the given point x.
     *
     * @param x The input vector at which to evaluate the log-likelihood.
     * @return The log-likelihood value at the given point.
     *
     * @note This method assumes that the input vector x has the same dimension as the Gaussian distribution.
     */
    virtual Scalar log(const Eigen::VectorX<Scalar> & x) const override
    {
        assert(x.cols() == 1);
        assert(x.size() == dim());

        // Compute log N(x; mu, P) where P = S.'*S
        // log N(x; mu, P) = -0.5*(x - mu).'*inv(P)*(x - mu) - 0.5*log(det(2*pi*P))

        // TODO: Numerically stable version
        const Eigen::Index n = dim();
        const Eigen::VectorX<Scalar> e = x - mu_;

        // y = S^{-T} e  (solve S^T y = e), so e^T P^{-1} e = ||y||^2
        const Eigen::VectorX<Scalar> y =
            S_.template triangularView<Eigen::Upper>().transpose().solve(e);
        const Scalar quad = y.squaredNorm(); // Quadratic form = e^T P^{-1} e = ||y||²

        using std::log;
        using std::abs;

        // 0.5 * log|P| = sum_i log(|S_ii|)
        // Slow way
        // Scalar half_logdetP = Scalar(0);
        // for (Eigen::Index i = 0; i < n; ++i) {
        //     half_logdetP += log(abs(S_(i, i)));
        // }
        // 0.5 * log|P| = sum_i log(|S_ii|)
        Scalar half_logdetP = (S_.diagonal().array().abs().log()).sum();


        const Scalar log2pi = log(Scalar(2) * Scalar(std::numbers::pi));
        // log N(x; μ, P) = -0.5*quad - 0.5*n*log(2π) - 0.5*log|P|
        // and 0.5*log|P| = half_logdetP
        // or // Formula (Lab eq. (6)):
        // log N = -0.5*||y||² - 0.5*n*log(2π) - sum log|S_ii|
        return Scalar(-0.5) * quad - Scalar(0.5) * Scalar(n) * log2pi - half_logdetP;

        // // Really bad version
        // Eigen::MatrixX<Scalar> P = S_.transpose()*S_;   // Bad, because unnecessary and loss of precision
        // Eigen::MatrixX<Scalar> Pinv = P.inverse();      // Bad, because you should know better (https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/)
        // Scalar quadraticForm = (x - mu_).transpose()*Pinv*(x - mu_);
        // using std::log, std::sqrt, std::exp;            // Bring selected math functions into global namespace
        // return log( 1.0/sqrt( (2*std::numbers::pi*P).determinant() )*exp(-0.5*quadraticForm) ); // Bad, because determinant, underflow, overflow and loss of precision
    }

    /**
     * @brief Compute the log-likelihood of a given point and its gradient under the Gaussian distribution.
     *
     * This method calculates the natural logarithm of the probability density function
     * of the Gaussian distribution at the given point x and computes its gradient.
     *
     * @param x The input vector at which to evaluate the log-likelihood.
     * @param g Reference to a vector where the gradient will be stored.
     * @return The log-likelihood value at the given point.
     *
     * @note This method assumes that the input vector x has the same dimension as the Gaussian distribution.
     *       The gradient vector g will be resized to match the dimension of x.
     */
    Scalar log(const Eigen::VectorX<Scalar> & x, Eigen::VectorX<Scalar> & g) const
    {
        // TODO: Compute gradient of log N(x; mu, P) w.r.t. x and write it to g
        // return log(x);
        assert(x.cols() == 1);
        assert(x.size() == dim());

        const Eigen::Index n = dim();
        const Eigen::VectorX<Scalar> e = x - mu_;

        // y = S^{-T} e
        const Eigen::VectorX<Scalar> y = S_.template triangularView<Eigen::Upper>().transpose().solve(e);
        // z = S^{-1} y = P^{-1} e
        const Eigen::VectorX<Scalar> z = S_.template triangularView<Eigen::Upper>().solve(y);

        // gradient of log N wrt x ∂/∂x logN = -P^{-1}(x - μ))
        g = -z;

        return log(x);
        }

    /**
     * @brief Compute the log-likelihood of a given point, its gradient, and Hessian under the Gaussian distribution.
     *
     * This method calculates the natural logarithm of the probability density function
     * of the Gaussian distribution at the given point x, computes its gradient, and Hessian matrix.
     *
     * @param x The input vector at which to evaluate the log-likelihood.
     * @param g Reference to a vector where the gradient will be stored.
     * @param H Reference to a matrix where the Hessian will be stored.
     * @return The log-likelihood value at the given point.
     *
     * @note This method assumes that the input vector x has the same dimension as the Gaussian distribution.
     *       The gradient vector g and Hessian matrix H will be resized to match the dimension of x.
     */
    Scalar log(const Eigen::VectorX<Scalar> & x, Eigen::VectorX<Scalar> & g, Eigen::MatrixX<Scalar> & H) const
    {
        // TODO: Compute Hessian of log N(x; mu, P) w.r.t. x and write it to H
        // return log(x, g);
        assert(x.cols() == 1);
        assert(x.size() == dim());

        const Eigen::Index n = dim();

        // Hessian: -P^{-1}
        // Solve S * X = I  ->  X = S^{-1}
        // Build stably: compute X = S^{-1}, then H = -X X^T
        Eigen::MatrixX<Scalar> X =
            S_.template triangularView<Eigen::Upper>().solve(
                Eigen::MatrixX<Scalar>::Identity(n, n)
            );
        H = -(X * X.transpose());

        return log(x, g);
    }

    /**
     * @brief Join two Gaussian distributions into a joint distribution.
     *
     * This method combines the current Gaussian distribution with another one,
     * creating a joint distribution. The resulting distribution represents
     * the joint probability of both input distributions, assuming they are independent.
     *
     * @param other The other Gaussian object to join with.
     * @return A new Gaussian object representing the joint distribution.
     */
    Gaussian join(const Gaussian & other) const
    {
        const Eigen::Index & n1 = dim();
        const Eigen::Index & n2 = other.dim();
        Gaussian out(n1 + n2);
        out.mu_ << mu_, other.mu_;
        out.S_ << S_,                                   Eigen::MatrixX<Scalar>::Zero(n1, n2),
                  Eigen::MatrixX<Scalar>::Zero(n2, n1), other.S_;
        return out;
    }

    /**
     * @brief Create a new Gaussian distribution by appending another Gaussian distribution.
     *
     * This operator combines the current Gaussian distribution with another one,
     * effectively creating a joint distribution of independent variables.
     * The resulting distribution will have a dimension equal to the sum of the
     * dimensions of the two input distributions.
     *
     * @param other The Gaussian distribution to append to this one.
     * @return A new Gaussian distribution representing the joint distribution.
     *
     * @note This operation assumes that the variables in the two distributions
     *       are independent, resulting in a block-diagonal covariance matrix.
     */
    Gaussian operator*(const Gaussian & other) const
    {
        return join(other);
    }

    /**
     * @brief Append another Gaussian distribution to this one.
     *
     * This operator combines the current Gaussian distribution with another one,
     * effectively creating a joint distribution of independent variables.
     * The resulting distribution will have a dimension equal to the sum of the
     * dimensions of the two input distributions.
     *
     * @param other The Gaussian distribution to append to this one.
     * @return A reference to the modified Gaussian distribution (*this).
     *
     * @note This operation assumes that the variables in the two distributions
     *       are independent, resulting in a block-diagonal covariance matrix.
     */
    Gaussian & operator*=(const Gaussian & other)
    {
        const Eigen::Index & n1 = dim();
        const Eigen::Index & n2 = other.dim();
        mu_.conservativeResize(n1 + n2);
        mu_.tail(n2) = other.mu_;
        S_.conservativeResizeLike(Eigen::MatrixX<Scalar>::Zero(n1 + n2, n1 + n2));
        S_.bottomRightCorner(n2, n2) = other.S_;
        return *this;
    }

    /**
     * @brief Check if a given point is within the confidence region of the Gaussian distribution.
     *
     * This method determines whether the input vector x is within the confidence region
     * defined by nSigma standard deviations from the mean of the Gaussian distribution.
     *
     * @param x The input vector to check.
     * @param nSigma The number of standard deviations defining the confidence region (default: 3.0).
     * @return True if the point is within the confidence region, false otherwise.
     */
    virtual bool isWithinConfidenceRegion(const Eigen::VectorX<Scalar> & x, double nSigma = 3.0) const override
    {
        const Eigen::Index & n = dim();
        // TODO
            assert(x.size() == n);

        // Mahalanobis^2 via triangular solves: r^2 = (x-μ)^T P^{-1} (x-μ) || S^{-T} (x−μ) ||^2
        const Eigen::VectorX<Scalar> e = x - mu_;
        const Eigen::VectorX<Scalar> y = S_.template triangularView<Eigen::Upper>().transpose().solve(e);
        const double r2 = static_cast<double>(y.squaredNorm());

        // Match 1D ±nSigma probability: p = 2*Φ(nSigma) - 1
        const double p = 2.0 * normcdf(static_cast<double>(nSigma)) - 1.0;

        // χ² threshold with ν = n / r² ≤ χ²_n(p)
        const double thr = chi2inv(p, static_cast<double>(n));

        return r2 <= thr;
    }

    /**
     * @brief Compute the quadric surface coefficients for a given number of standard deviations.
     *
     * This method calculates the coefficients of the quadric surface that represents
     * the confidence ellipsoid of the 3D Gaussian distribution. The surface is defined
     * for a specified number of standard deviations (nSigma).
     *
     * @param nSigma The number of standard deviations to use for the confidence ellipsoid (default: 3.0).
     * @return A 4x4 matrix containing the quadric surface coefficients.
     *
     * @note This method assumes that the Gaussian distribution is three-dimensional.
     */
    Eigen::Matrix4<Scalar> quadricSurface(double nSigma = 3.0) const
    {
        const Eigen::Index & n = dim();
        assert(n == 3);
        
        Eigen::Matrix4<Scalar> Q;
        // TODO
        // A = P^{-1} using the stable infoMat() (no explicit inverse of P)
        const Eigen::MatrixX<Scalar> A = infoMat();     // 3x3
        const Eigen::VectorX<Scalar> mu = mean();       // 3x1
        const Eigen::VectorX<Scalar> k  = A * mu;       // 3x1 ((used for the off-diagonal blocks))

        // Probability that corresponds to "± nSigma" in 1D
        const double p  = 2.0 * normcdf(nSigma) - 1.0;
        // Chi-square threshold for 3 DoF
        const double c3 = chi2inv(p, 3.0); // 3 for 3 dof

        Q.setZero();
        Q.template topLeftCorner<3,3>() = A;
        Q.template topRightCorner<3,1>() = -k;
        Q.template bottomLeftCorner<1,3>() = -k.transpose();
        Q(3,3) = static_cast<Scalar>(mu.dot(k) - c3);

        return Q;
    }

protected:
    Eigen::VectorX<Scalar> mu_;  ///< The mean vector of the Gaussian distribution.
    Eigen::MatrixX<Scalar> S_;   ///< The square root of the covariance matrix (upper triangular).
};

#endif
