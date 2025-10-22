#ifndef SYSTEMSLAM_H
#define SYSTEMSLAM_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "Camera.h"
#include "SystemEstimator.h"

/*
 * State contains body velocities, body pose and landmark states
 *
 *     [ vBNb     ]  Body translational velocity (body-fixed)
 *     [ omegaBNb ]  Body angular velocity (body-fixed)
 * x = [ rBNn     ]  Body position (world-fixed)
 *     [ Thetanb  ]  Body orientation (world-fixed)
 *     [ m        ]  Landmark map states (undefined in this class)
 *
 */
class SystemSLAM : public SystemEstimator
{
public:
    explicit SystemSLAM(const GaussianInfo<double> & density);
    virtual SystemSLAM * clone() const = 0;
    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const override;
    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const override;
    virtual Eigen::VectorXd input(double t, const Eigen::VectorXd & x) const override;
    virtual GaussianInfo<double> processNoiseDensity(double dt) const override;
    virtual std::vector<Eigen::Index> processNoiseIndex() const override;

    virtual GaussianInfo<double> bodyPositionDensity() const;
    virtual GaussianInfo<double> bodyOrientationDensity() const;
    virtual GaussianInfo<double> bodyTranslationalVelocityDensity() const;
    virtual GaussianInfo<double> bodyAngularVelocityDensity() const;

    template <typename Scalar> static Eigen::Vector3<Scalar> cameraPosition(const Camera & cam, const Eigen::VectorX<Scalar> & x);
    static Eigen::Vector3d cameraPosition(const Camera & cam, const Eigen::VectorXd & x, Eigen::MatrixXd & J);
    template <typename Scalar> static Eigen::Matrix3<Scalar> cameraOrientation(const Camera & cam, const Eigen::VectorX<Scalar> & x);
    template <typename Scalar> static Eigen::Vector3<Scalar> cameraOrientationEuler(const Camera & cam, const Eigen::VectorX<Scalar> & x);
    static Eigen::Vector3d cameraOrientationEuler(const Camera & cam, const Eigen::VectorXd & x, Eigen::MatrixXd & J);

    virtual GaussianInfo<double> cameraPositionDensity(const Camera & cam) const;
    virtual GaussianInfo<double> cameraOrientationEulerDensity(const Camera & cam) const;

    virtual std::size_t numberLandmarks() const = 0;
    virtual GaussianInfo<double> landmarkPositionDensity(std::size_t idxLandmark) const;
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const = 0;

    cv::Mat & view();
    const cv::Mat & view() const;
protected:
    cv::Mat view_;
};

#include "rotation.hpp"
#include "Pose.hpp"
#include <opencv2/core/eigen.hpp>

template <typename Scalar>
Eigen::Vector3<Scalar> SystemSLAM::cameraPosition(const Camera & camera, const Eigen::VectorX<Scalar> & x)
{
    Pose<Scalar> Tnb;
    Tnb.rotationMatrix = rpy2rot(x.template segment<3>(9)); // Rnb
    Tnb.translationVector = x.template segment<3>(6);       // rBNn
    Pose<Scalar> Tnc = camera.bodyToCamera(Tnb);
    return Tnc.translationVector;                           // rCNn
}

template <typename Scalar>
Eigen::Matrix3<Scalar> SystemSLAM::cameraOrientation(const Camera & camera, const Eigen::VectorX<Scalar> & x)
{
    Pose<Scalar> Tnb;
    Tnb.rotationMatrix = rpy2rot(x.template segment<3>(9)); // Rnb
    Tnb.translationVector = x.template segment<3>(6);       // rBNn
    Pose<Scalar> Tnc = camera.bodyToCamera(Tnb);
    return Tnc.rotationMatrix;                              // Rnc
}

template <typename Scalar>
Eigen::Vector3<Scalar> SystemSLAM::cameraOrientationEuler(const Camera & camera, const Eigen::VectorX<Scalar> & x)
{
    return rot2rpy(cameraOrientation(camera, x));
}

#endif
