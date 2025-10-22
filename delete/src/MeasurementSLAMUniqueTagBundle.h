#pragma once

#include <vector>
#include <Eigen/Core>
#include "MeasurementSLAMPointBundle.h"

/*
Scenario 1: Unique ArUco tags as pose landmarks.
State per landmark j: m_j = [ r^n_{j/N} (3), Θ^n_j (3) ]^T   (assignment Eq. (6)).
Each measurement aggregates the 4 image-plane corners of a detected tag
y_i = [u1,v1, u2,v2, u3,v3, u4,v4]^T, with object-space corners from (8)-(9)
using edge length ℓ = 0.166 m. Log-likelihood uses per-corner Gaussian terms
and a missed-detection penalty −4|U| log|Y| (Eq. (7)).
*/
class MeasurementSLAMUniqueTagBundle : public MeasurementPointBundle
{
public:
    /**
     * @param time   Measurement timestamp
     * @param Y      Corner bundle, shape 2×(4N): columns [u1 v1 u2 v2 u3 v3 u4 v4]ᵀ per tag
     * @param camera Camera model (intrinsics + distortion + body↔camera extrinsics)
     * @param ids    Tag IDs (size N), used for trivial data association
     */
    MeasurementSLAMUniqueTagBundle(double time,
                                   const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                   const Camera& camera,
                                   const std::vector<int>& ids);

    /// Clone preserving dynamic type (required by plotting/estimation pipeline).
    MeasurementSLAM* clone() const override;

    // Persistent ID mapping: landmark index → tag ID (−1 if uninitialized)
    void setIdByLandmark(const std::vector<int>& m) { id_by_landmark_ = m; }
    const std::vector<int>& idByLandmark() const { return id_by_landmark_; }

    /// True iff landmark j was associated to a detection this frame (blue in plot).
    bool isEffectivelyAssociated(std::size_t landmarkIdx) const;

    /**
     * ID-based association + visibility tagging.
     * - Association: tag ID → feature index (trivial, unique IDs).
     * - Visibility (|U| term in (7)): uses PRIOR MEAN state to decide if all 4
     *   corners would fall strictly inside the image (conservative check).
     */
    const std::vector<int>& associate(const SystemSLAM& system,
                                      const std::vector<std::size_t>& idxLandmarks) override;

    /**
     * Measurement update for Scenario 1.
     * Applies the Kalman/optimizer update; no landmark deletion (loop-closure).
     */
    void update(SystemBase& system) override;

    // Log-likelihood overloads:
    //   scalar:      −½ σ⁻² ∑‖y−h(x)‖² − 4|U| log|Y|
    //   + gradient:   ∑ σ⁻² Jᵀ r
    //   + Hessian:   −∑ σ⁻² Jᵀ J  (Gauss–Newton); |U| term has zero ∇, zero H.
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system) const override;
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g) const override;
    double logLikelihood(const Eigen::VectorXd& x, const SystemEstimator& system, Eigen::VectorXd& g, Eigen::MatrixXd& H) const override;

    const std::vector<bool>& isVisible() const { return is_visible_; }

protected:
    /**
     * Predicts the 4 tag-corner pixel locations for landmark j:
     *   1) Tag corners in tag frame from (9) with ℓ = TAG_SIZE.
     *   2) World corners r^n_{jc} = R^n_L r^L_{jc} + r^n_L  (from (8)).
     *   3) Camera frame r^c_{jc} = R^c_n (r^n_{jc} − r^n_C).
     *   4) Projection u = π(K,R,t,r^c_{jc}) via Camera::vectorToPixel.
     * Returns h_j(x) = [u1,v1, u2,v2, u3,v3, u4,v4]^T.
     */
    Eigen::Matrix<double,8,1> predictTagCorners(const Eigen::VectorXd& x,
                                                const SystemSLAM& system,
                                                std::size_t idxLandmark) const;

    /// Autodiff-friendly overload; Scalar ∈ {double, autodiff::dual}.
    template<typename Scalar>
    Eigen::Matrix<Scalar,8,1> predictTagCornersT(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
                                                  const SystemSLAM& system,
                                                  std::size_t idxLandmark) const;

private:
    std::vector<int>  ids_;                     // Detected tag IDs (size N)
    std::vector<int>  id_by_landmark_;          // Persistent mapping landmark→ID
    std::vector<bool> is_visible_;              // Visibility at prior mean (|U| in (7))
    std::vector<bool> is_effectively_associated_; // Associated this frame

    static constexpr double TAG_SIZE = 0.166;   // Tag edge length ℓ [m] (assignment)
};
