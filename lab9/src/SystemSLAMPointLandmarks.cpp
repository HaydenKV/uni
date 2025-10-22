#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"
#include "SystemSLAMPointLandmarks.h"

SystemSLAMPointLandmarks::SystemSLAMPointLandmarks(const GaussianInfo<double> & density)
    : SystemSLAM(density)
{

}

SystemSLAM * SystemSLAMPointLandmarks::clone() const
{
    return new SystemSLAMPointLandmarks(*this);
}

std::size_t SystemSLAMPointLandmarks::numberLandmarks() const
{
    return (density.dim() - 12)/3;
}

std::size_t SystemSLAMPointLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 3*idxLandmark;    
}