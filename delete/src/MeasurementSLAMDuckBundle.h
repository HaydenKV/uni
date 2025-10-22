#ifndef MEASUREMENTSLAMDUCKBUNDLE_H
#define MEASUREMENTSLAMDUCKBUNDLE_H

#include <Eigen/Core>
#include <vector>
#include "MeasurementSLAM.h"
#include "Camera.h"
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"

class MeasurementSLAMDuckBundle : public MeasurementSLAM
{
public:
    MeasurementSLAMDuckBundle(double time,
                              const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                              const Eigen::VectorXd& A,
                              const Camera& camera,
                              double sigma_c_px,
                              double sigma_a_px2,
                              double duck_radius_m);

    MeasurementSLAM* clone() const override;

    // Base Measurement interface
    Eigen::VectorXd simulate(const Eigen::VectorXd& x, const SystemEstimator& system) const override;

    // Likelihoods over stacked [u v A ...] for associated pairs
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system) const override;
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g) const override;
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g, Eigen::MatrixXd& H) const override;

    // Plot needs this to colour landmarks blue when matched this frame
    bool isEffectivelyAssociated(std::size_t j) const {
        return j < is_effectively_associated_.size() && is_effectively_associated_[j];
    }

    // Densities required by base API (used by Plot for 2D ellipses)
    GaussianInfo<double> predictFeatureDensity(const SystemSLAM& system, std::size_t idxLandmark) const override;
    GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM& system, const std::vector<std::size_t>& idxLandmarks) const override;
    
    // Expose centroids for SNN association (shape 2×N)
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Y() const { return Yuv_; }

    // Association uses 2D (u,v) density; pass that to SNN
    const std::vector<int>& associate(const SystemSLAM& system, const std::vector<std::size_t>& idxLandmarks) override;

    // Association results (landmark index -> feature index or -1)
    const std::vector<int>& idxFeatures() const { return idxFeatures_; }

protected:
    void update(SystemBase& system) override;

    // Per-landmark predicted (u,v,A)
    Eigen::Vector3d predictDuck(const Eigen::VectorXd& x, Eigen::MatrixXd& J, const SystemSLAM& system, std::size_t idxLandmark) const;

    // CORRECTED: predictDuckT is no longer a template method
    Eigen::Vector3d predictDuckT(const Eigen::VectorXd& x,
                                 const SystemSLAM& system,
                                 std::size_t idxLandmark) const;

    // Bundle prediction (3× per landmark): stacks [u v A] for all landmarks
    Eigen::VectorXd predictDuckBundle(const Eigen::VectorXd& x, Eigen::MatrixXd& J, const SystemSLAM& system,
                                      const std::vector<std::size_t>& idxLandmarks) const;

private:
    // ---- Helpers (private) ----
    GaussianInfo<double> predictDuckBundleDensity(const SystemSLAM& system,
                                                  const std::vector<std::size_t>& idxLandmarks) const;

    GaussianInfo<double> predictCentroidBundleDensity(const SystemSLAM& system,
                                                      const std::vector<std::size_t>& idxLandmarks) const;

    // ---- Measurement storage ----
    Eigen::Matrix<double,2,Eigen::Dynamic> Yuv_;  // centroids (2×N) for this frame
    Eigen::VectorXd                        A_;      // areas (N)

    // ---- Noise/tuning ----
    double sigma_c_;    // pixel std for (u,v)
    double sigma_a_;    // area std (pixels^2)
    double duck_r_m_;   // physical duck radius [m]
    double fx_, fy_;    // cached intrinsics for area model

    // ---- Association state ----
    std::vector<int>  idxFeatures_;               // LM j -> feature index (-1 if none)
    std::vector<bool> is_effectively_associated_;   // for Plot colouring
};

#endif // MEASUREMENTSLAMDUCKBUNDLE_H