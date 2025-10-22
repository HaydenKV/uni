/**
 * @file DensityBase.hpp
 * @brief Defines the base class for probability density functions.
 */

#include <cmath>
#include <Eigen/Core>

/**
 * @brief Base class for probability density functions.
 * 
 * @tparam Scalar The scalar type used for calculations (default: double).
 */
template <typename Scalar = double>
class DensityBase
{
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~DensityBase() = default;

    /**
     * @brief Computes the log of the probability density function.
     * 
     * @param x The input vector.
     * @return The log of the probability density at x.
     */
    virtual Scalar log(const Eigen::VectorX<Scalar> & x) const = 0;
    
    /**
     * @brief Evaluates the probability density function.
     * 
     * @param x The input vector.
     * @return The probability density at x.
     */
    Scalar eval(const Eigen::VectorX<Scalar> & x) const
    {
        using std::exp;
        return exp(log(x));
    }
};