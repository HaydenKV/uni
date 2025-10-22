#include "Camera.h"
#include "MeasurementSLAM.h"

MeasurementSLAM::MeasurementSLAM(double time, const Camera & camera)
    : Measurement(time)
    , camera_(camera)
{}

MeasurementSLAM::~MeasurementSLAM() = default;