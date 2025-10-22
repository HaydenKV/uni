#ifndef SYSTEMSLAMPOINTLANDMARKS_H
#define SYSTEMSLAMPOINTLANDMARKS_H

#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"

/*
 * State containing body velocities, body pose and landmark positions
 *
 * [ vBNb     ]  Body translational velocity (body-fixed)
 * [ omegaBNb ]  Body angular velocity (body-fixed)
 * [ rBNn     ]  Body position (world-fixed)
 * x = [ Thetanb  ]  Body orientation (world-fixed, Euler rpy)
 * [ rL1Nn    ]  Landmark 1 position (world-fixed)
 * [ rL2Nn    ]  Landmark 2 position (world-fixed)
 * [ ...      ]  ...
 *
 */
class SystemSLAMPointLandmarks : public SystemSLAM
{
public:
    explicit SystemSLAMPointLandmarks(const GaussianInfo<double> & density);
    SystemSLAM * clone() const override;
    virtual std::size_t numberLandmarks() const override;
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const override;

    // Append a single 3-D point landmark (position only), returns its index.
    std::size_t appendLandmark(const Eigen::Vector3d& rLNn,
                               const Eigen::Matrix3d& Spos);

    // Init new point landmarks from duck detections (centroids+areas).
    // Area model: A = (fx*fy*pi*r^2)/depth^2  => depth = sqrt(fx*fy*pi*r^2 / A)
    std::size_t appendFromDuckDetections(const Camera& cam,
                                         const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                                         const Eigen::VectorXd& A,
                                         double fx, double fy,
                                         double duck_r_m,
                                         double pos_sigma_m);

    // Miss counter and culling utilities.
    void updateFailureCounter(std::size_t j, bool incrementIfFailed);
    void cullFailed(int threshold = 50);

private:
    std::vector<int> failCount_;
};

#endif
