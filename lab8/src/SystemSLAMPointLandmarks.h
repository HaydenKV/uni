#ifndef SYSTEMSLAMPOINTLANDMARKS_H
#define SYSTEMSLAMPOINTLANDMARKS_H

#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"

/*
 * State containing body velocities, body pose and landmark positions
 *
 *     [ vBNb     ]  Body translational velocity (body-fixed)
 *     [ omegaBNb ]  Body angular velocity (body-fixed)
 *     [ rBNn     ]  Body position (world-fixed)
 * x = [ Thetanb  ]  Body orientation (world-fixed)
 *     [ rL1Nn    ]  Landmark 1 position (world-fixed)
 *     [ rL2Nn    ]  Landmark 2 position (world-fixed)
 *     [ ...      ]  ...
 *
 */
class SystemSLAMPointLandmarks : public SystemSLAM
{
public:
    explicit SystemSLAMPointLandmarks(const GaussianInfo<double> & density);
    SystemSLAM * clone() const override;
    virtual std::size_t numberLandmarks() const override;
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const override;
};

#endif