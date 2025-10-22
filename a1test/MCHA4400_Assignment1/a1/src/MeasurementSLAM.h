#ifndef MEASUREMENTSLAM_H
#define MEASUREMENTSLAM_H

#include <cstddef>
#include <vector>
#include "Camera.h"
#include "SystemSLAM.h"
#include "Measurement.h"

class MeasurementSLAM : public Measurement
{
public:
    MeasurementSLAM(double time, const Camera & camera);
    virtual MeasurementSLAM * clone() const = 0;
    virtual ~MeasurementSLAM() override;

    virtual GaussianInfo<double> predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const = 0;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const = 0;
    virtual const std::vector<int> & associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) = 0;

protected:
    const Camera & camera_;
};

#endif