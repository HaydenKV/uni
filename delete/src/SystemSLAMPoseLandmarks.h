#ifndef SYSTEMSLAMPOSELANDMARKS_H
#define SYSTEMSLAMPOSELANDMARKS_H

#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"

/*
 * State containing body velocities, body pose and landmark poses
 *
 *     [ vBNb     ]  Body translational velocity (body-fixed)
 *     [ omegaBNb ]  Body angular velocity (body-fixed)
 *     [ rBNn     ]  Body position (world-fixed)
 *     [ Thetanb  ]  Body orientation (world-fixed)
 * x = [ rL1Nn     ]  Landmark 1 position (world-fixed)
 *     [ omegaL1Nc ]  Landmark 1 orientation (world-fixed)
 *     [ rL2Nn     ]  Landmark 2 position (world-fixed)
 *     [ omegaL2Nc ]  Landmark 2 orientation (world-fixed)
 *     [ ...       ]  ...
 *
 */
class SystemSLAMPoseLandmarks : public SystemSLAM
{
public:
    explicit SystemSLAMPoseLandmarks(const GaussianInfo<double> & density);
    SystemSLAM * clone() const override;
    virtual std::size_t numberLandmarks() const override;
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const override;

    // ===============================
    // SystemSLAMPoseLandmarks::appendLandmark
    // ===============================
    // Appends a 6-DOF pose landmark to the SLAM state: [r_nj; Theta_nj] (position in world, Euler angles).
    // Inputs:
    //   rnj      : Eigen::Vector3d  (world position of tag/landmark)
    //   Thetanj  : Eigen::Vector3d  (landmark world orientation as Euler rpy [rad])
    //   Sj       : 6x6 upper-triangular sqrt-covariance for the landmark block
    // Returns:
    //   size_t index of the newly appended landmark (0-based among the pose landmarks).
    //
    // Notes:
    // - We expand the state's mean and sqrt-covariance by 6.
    // - Cross-covariance between existing states and the new landmark is initialized to zero
    //   (you can later add a data-driven cross term if you propagate uncertainties here).
    // - Sj should be reasonably conservative; SLAM updates will tighten it.
    std::size_t appendLandmark(const Eigen::Vector3d& rnj,
                               const Eigen::Vector3d& Thetanj,
                               const Eigen::Matrix<double,6,6>& Sj);
};

#endif