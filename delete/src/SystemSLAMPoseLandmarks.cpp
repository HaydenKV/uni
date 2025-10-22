#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"
#include "SystemSLAMPoseLandmarks.h"
#include <Eigen/Cholesky>

SystemSLAMPoseLandmarks::SystemSLAMPoseLandmarks(const GaussianInfo<double> & density)
    : SystemSLAM(density)
{

}

SystemSLAM * SystemSLAMPoseLandmarks::clone() const
{
    return new SystemSLAMPoseLandmarks(*this);
}

std::size_t SystemSLAMPoseLandmarks::numberLandmarks() const
{
    return (density.dim() - 12)/6;
}

std::size_t SystemSLAMPoseLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 6*idxLandmark;    
}

/*
Append a 6-DOF pose landmark m_j = [ r^n_{j/N}, Θ^n_j ]^T (assignment Eq. (6))
to the SLAM state by expanding the mean and upper-triangular square-root
covariance:

    x_new = [ x_old; r^n_{j/N}; Θ^n_j ],
    S_new = [ S_old     0;
               0       S_j ].

Landmark dynamics are static in (4)–(5), making zero initial cross-blocks
consistent; cross-covariances are introduced by subsequent measurement updates.
S_j must be upper-triangular so that S_new remains a valid square-root factor.
Returns the 0-based index of the appended landmark.
*/
std::size_t SystemSLAMPoseLandmarks::appendLandmark(
    const Eigen::Vector3d& rnj,
    const Eigen::Vector3d& Thetanj,
    const Eigen::Matrix<double,6,6>& Sj
)
{
    // Current state
    const Eigen::VectorXd mu_old = density.mean();
    const Eigen::MatrixXd S_old  = density.sqrtCov();
    const int n_old = static_cast<int>(mu_old.size());

    // New sizes (+6 for pose landmark)
    const int n_new = n_old + 6;

    // New mean: append [r^n_{j/N}; Θ^n_j]
    Eigen::VectorXd mu_new(n_new);
    mu_new.head(n_old) = mu_old;
    mu_new.segment<3>(n_old + 0) = rnj;      // position (world frame)
    mu_new.segment<3>(n_old + 3) = Thetanj;  // Euler rpy (Θ^n_j)

    // New sqrt-covariance (upper-triangular by construction)
    Eigen::MatrixXd S_new = Eigen::MatrixXd::Zero(n_new, n_new);
    S_new.topLeftCorner(n_old, n_old) = S_old;  // existing block
    S_new.block(n_old, n_old, 6, 6) = Sj;       // landmark block

    // Cross blocks remain zero initially (static landmark model in (4)–(5)).

    // Reconstruct density from sqrt moments
    density = GaussianInfo<double>::fromSqrtMoment(mu_new, S_new);

    // Index among pose landmarks (contiguous after 12 body states)
    const std::size_t lm_index = static_cast<std::size_t>((n_new - 12) / 6 - 1);
    return lm_index;
}
