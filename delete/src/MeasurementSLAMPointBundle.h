#ifndef MEASUREMENTSLAMPOINTBUNDLE_H
#define MEASUREMENTSLAMPOINTBUNDLE_H

#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementSLAM.h"

class MeasurementPointBundle : public MeasurementSLAM
{
public:
    MeasurementPointBundle(double time, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera);
    MeasurementSLAM * clone() const override;
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    template <typename Scalar> Eigen::Vector2<Scalar> predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const;
    Eigen::Vector2d predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const;
    virtual GaussianInfo<double> predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const override;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const override;

    virtual const std::vector<int> & associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) override;

    const std::vector<int>& idxFeatures() const { return idxFeatures_; }
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Y() const { return Y_; }
protected:
    virtual void update(SystemBase & system) override;
    Eigen::Matrix<double, 2, Eigen::Dynamic> Y_;    // Feature bundle
    double sigma_;                                  // Feature error standard deviation (in pixels)
    std::vector<int> idxFeatures_;                  // Features associated with visible landmarks
};

// Image feature location for a given landmark
template <typename Scalar>
Eigen::Vector2<Scalar> MeasurementPointBundle::predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Obtain camera pose from state
    Pose<Scalar> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = system.cameraOrientation(camera_, x); // Rnc

    // Obtain landmark position from state
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rPNn = x.template segment<3>(idx);

    // Camera vector
    // Transform nav → camera: rPCc = Rcn * (rPNn - rCNn), with Rcn = Rncᵀ
    const Eigen::Matrix3<Scalar> Rcn = Tnc.rotationMatrix.transpose();
    Eigen::Vector3<Scalar> rPCc = Rcn * (rPNn - Tnc.translationVector);
    // TODO: Lab 8

    // Pixel coordinates
    Eigen::Vector2<Scalar> rQOi = camera_.vectorToPixel(rPCc);
    // TODO: Lab 8
    return rQOi;
}

// Image feature locations for a bundle of landmarks
template <typename Scalar>
Eigen::VectorX<Scalar> MeasurementPointBundle::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    assert(x.size() == system.density.dim());

    const std::size_t & nL = idxLandmarks.size();
    Eigen::VectorX<Scalar> h(2*nL);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::Vector2<Scalar> rQOi = predictFeature(x, system, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 9
        h.template segment<2>(2*i) = rQOi;
    }
    return h;
}

#endif
