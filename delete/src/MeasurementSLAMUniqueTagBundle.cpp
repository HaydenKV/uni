#include <cstddef>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "SystemSLAMPoseLandmarks.h"
#include "Camera.h"
#include "Measurement.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "Pose.hpp"
#include "rotation.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

namespace {
    // Thread-local scratch for current evaluation chain (avoids rebuild in scalar/grad/Hess)
    thread_local bool tl_assoc_ready = false;
    thread_local std::vector<std::size_t> tl_idxUseLandmarks;  // selected landmarks
    thread_local std::vector<int>         tl_idxUseFeatures;   // matched feature indices

    /*
    Build (landmark, feature) selection exactly once per evaluation chain.
    Y: 2×(4N) stacks 4 corners per tag; idxFeatures holds tag-index (0..N−1) or −1.
    */
    inline void ensureAssociatedOnce(const SystemSLAM& sys,
                                     const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                     const std::vector<int>& idxFeatures)
    {
        if (tl_assoc_ready) return;

        tl_idxUseLandmarks.clear();
        tl_idxUseFeatures.clear();
        tl_idxUseLandmarks.reserve(sys.numberLandmarks());
        tl_idxUseFeatures.reserve(sys.numberLandmarks());

        // N tags ⇒ 4N corner columns (guard: multiple of 4).
        const int N = static_cast<int>(Y.cols()) / 4;
        assert(Y.cols() % 4 == 0 && "Y must have 4 columns per tag");

        const std::size_t nL = sys.numberLandmarks();
        for (std::size_t j = 0; j < nL; ++j)
        {
            if (j < idxFeatures.size()) {
                const int fi = idxFeatures[j];
                if (fi >= 0 && fi < N) { // compare to N (tags), not 4N (columns)
                    tl_idxUseLandmarks.push_back(j);
                    tl_idxUseFeatures.push_back(fi);
                }
            }
        }
        tl_assoc_ready = true;
    }

    inline void clearAssociationScratch() {
        tl_assoc_ready = false;
        tl_idxUseLandmarks.clear();
        tl_idxUseFeatures.clear();
    }
} // namespace

/*
Constructor
- Validates that Y is 2×(4N) (each tag contributes four corner columns).
- Sets σ (image noise) and the trust-region Newton method used by the base class.
References: scenario measurement structure (7)-(9).
*/
MeasurementSLAMUniqueTagBundle::MeasurementSLAMUniqueTagBundle(
    double time,
    const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
    const Camera& camera,
    const std::vector<int>& ids)
: MeasurementPointBundle(time, Y, camera)
, ids_(ids)
, id_by_landmark_()
{
    assert(Y_.cols() % 4 == 0 && "Y packing error: must have 4 columns per tag detection");
    sigma_ = 2.0;
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

/*
Clone (preserves dynamic type and persistent association flags).
*/
MeasurementSLAM* MeasurementSLAMUniqueTagBundle::clone() const
{
    auto* copy = new MeasurementSLAMUniqueTagBundle(time_, Y_, camera_, ids_);
    copy->id_by_landmark_ = this->id_by_landmark_;
    copy->idxFeatures_ = this->idxFeatures_;
    copy->is_visible_ = this->is_visible_;
    copy->is_effectively_associated_ = this->is_effectively_associated_;
    copy->sigma_ = this->sigma_;
    return copy;
}

/*
Used by Plot: blue if effectively associated this frame, else red (if visible).
*/
bool MeasurementSLAMUniqueTagBundle::isEffectivelyAssociated(std::size_t landmarkIdx) const
{
    if (landmarkIdx >= is_effectively_associated_.size()) {
        return false;
    }
    return is_effectively_associated_[landmarkIdx];
}

/*
Association + visibility tagging.
- Association (ID→feature): trivial because tag IDs are unique (Eq. (6)).
- Visibility (|U| in (7)): using PRIOR MEAN state (piecewise-constant in x)
  and conservative image-bound test on all 4 corners (Eqs. (8)-(9) + Camera π()).
Color logic (visualization spec):
  blue : visible + associated
  red  : visible + unassociated (contributes to |U|)
  yellow: not visible (outside FOV)
*/
const std::vector<int>& MeasurementSLAMUniqueTagBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& /*idxLandmarks*/)
{
    const SystemSLAMPoseLandmarks& sysPose = dynamic_cast<const SystemSLAMPoseLandmarks&>(system);
    const std::size_t nL = sysPose.numberLandmarks();

    // Ensure persistent vectors sized to number of landmarks
    if (id_by_landmark_.size() < nL) {
        id_by_landmark_.resize(nL, -1);
    }

    // Reset per-frame
    is_visible_.assign(nL, false);
    is_effectively_associated_.assign(nL, false);
    idxFeatures_.assign(nL, -1);

    // Map: tag ID → feature index in Y (0..N−1)
    std::unordered_map<int, int> id2feat;
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        id2feat[ids_[i]] = static_cast<int>(i);
    }

    // Use PRIOR MEAN x̄ for visibility (keeps |U| independent of optimization variable)
    const Eigen::VectorXd xmean = sysPose.density.mean();

    for (std::size_t j = 0; j < nL; ++j)
    {
        const int tagId = id_by_landmark_[j];
        if (tagId < 0) continue; // landmark not yet assigned a real tag ID

        // 1) ID-based association (robust to state errors; quality gated upstream)
        if (auto it = id2feat.find(tagId); it != id2feat.end()) {
            const int featIdx = it->second;
            idxFeatures_[j] = featIdx;
            is_effectively_associated_[j] = true;
        }

        // 2) Predicted visibility for |U| term: all 4 corners strictly inside image
        const Eigen::Matrix<double,8,1> corners = predictTagCorners(xmean, sysPose, j);
        const bool allCornersVisible = camera_.areCornersInside(corners);
        is_visible_[j] = allCornersVisible;
    }

    return idxFeatures_;
}

/*
Scenario 1 update: perform measurement update; do NOT delete landmarks
(loop-closure opportunities must persist).
*/
void MeasurementSLAMUniqueTagBundle::update(SystemBase& system)
{
    SystemSLAMPoseLandmarks& systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks&>(system);

    // Establish associations for this frame
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
    associate(systemSLAM, idxLandmarks);

    // Kalman/optimizer update through base
    Measurement::update(system);
}

/*
Predicts tag corner pixels h_j(x) for landmark j.
Equations used:
  (8) r^n_{jc} = R^n_L(Θ^n_L) r^L_{jc} + r^n_{L/N}, with r^L_{jc} from (9), ℓ=TAG_SIZE.
  Camera mapping: r^c_{jc} = R^c_n (r^n_{jc} − r^n_{C/N}), then u=π(K,dist, r^c_{jc}).
*/
Eigen::Matrix<double,8,1> MeasurementSLAMUniqueTagBundle::predictTagCorners(
    const Eigen::VectorXd& x,
    const SystemSLAM& system,
    std::size_t idxLandmark) const
{
    return predictTagCornersT<double>(x, system, idxLandmark);
}

// Templated (double / autodiff::dual) version for use in autodiff Jacobians.
template<typename Scalar>
Eigen::Matrix<Scalar,8,1> MeasurementSLAMUniqueTagBundle::predictTagCornersT(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
    const SystemSLAM& system,
    std::size_t idxLandmark) const
{
    // Landmark pose: [ r^n_{L/N} (3), Θ^n_L (3) ] (Eq. (6))
    const std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    const Eigen::Matrix<Scalar,3,1> rLNn    = x.template segment<3>(idx);
    const Eigen::Matrix<Scalar,3,1> ThetaLn = x.template segment<3>(idx + 3);

    // R^n_L from roll-pitch-yaw (consistent with assignment conventions)
    const Eigen::Matrix<Scalar,3,3> RnL = rpy2rot(ThetaLn);

    // Body pose (B) from state; then compose with known body→camera (B→C)
    Pose<Scalar> Tnb;
    Tnb.translationVector = SystemSLAM::cameraPosition(camera_, x);   // r^n_{B/N}
    Tnb.rotationMatrix    = SystemSLAM::cameraOrientation(camera_, x);// R^n_B

    // T^n_C = T^n_B ∘ T^B_C (Camera in world)
    const Pose<Scalar> Tnc = camera_.bodyToCamera(Tnb);

    // Camera frame transforms
    const Eigen::Matrix<Scalar,3,3> Rcn = Tnc.rotationMatrix.transpose(); // R^c_n
    const Eigen::Matrix<Scalar,3,1> rCNn = Tnc.translationVector;        // r^n_{C/N}

    // Tag-frame corners r^L_{jc} (Eq. (9), ℓ = TAG_SIZE) in TL,TR,BR,BL order
    const Scalar half = static_cast<Scalar>(TAG_SIZE / 2.0);
    Eigen::Matrix<Scalar,3,4> cornersL;
    cornersL.col(0) << -half,  half, Scalar(0);
    cornersL.col(1) <<  half,  half, Scalar(0);
    cornersL.col(2) <<  half, -half, Scalar(0);
    cornersL.col(3) << -half, -half, Scalar(0);

    // Project to pixels
    Eigen::Matrix<Scalar,8,1> h; // [u1,v1, u2,v2, u3,v3, u4,v4]^T
    for (int c = 0; c < 4; ++c)
    {
        // World corner (8)
        const Eigen::Matrix<Scalar,3,1> rJcNn = rLNn + RnL * cornersL.col(c);
        // Camera frame
        const Eigen::Matrix<Scalar,3,1> rJcCc = Rcn * (rJcNn - rCNn);
        // Project using calibrated camera model
        const Eigen::Matrix<Scalar,2,1> uv = camera_.vectorToPixel(rJcCc);

        h(2*c)     = uv(0);
        h(2*c + 1) = uv(1);
    }
    return h;
}

/*
Log-likelihood (scalar)
  L(x) = ∑_{(i,j)∈A} −½σ⁻² || y_i − h_j(x) ||²  − 4|U| log|Y|.
- The −4 arises because each tag contributes 4 corners (Eq. (7)).
- |U| counts visible (by prior mean) but unassociated landmarks; independent of x.
*/
double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);

    ensureAssociatedOnce(sys, Y_, idxFeatures_);
    if (tl_idxUseLandmarks.empty()) {
        clearAssociationScratch();
        return 0.0;
    }

    const std::size_t k = tl_idxUseLandmarks.size();
    const double inv_R = 1.0 / (sigma_ * sigma_);
    double logL = 0.0;

    // Associated terms
    for (std::size_t i = 0; i < k; ++i)
    {
        const std::size_t j = tl_idxUseLandmarks[i];
        const int fi = tl_idxUseFeatures[i];

        // y_i: stack the 4 columns for tag fi
        Eigen::Matrix<double,8,1> yi;
        for (int c = 0; c < 4; ++c)
            yi.segment<2>(2*c) = Y_.col(4*fi + c);

        // h_j(x): projected corners
        const Eigen::Matrix<double,8,1> hi = predictTagCorners(x, sys, j);

        const Eigen::Matrix<double,8,1> ri = yi - hi;
        logL += -0.5 * inv_R * ri.squaredNorm();
    }

    // Missed-detection penalty −4|U| log|Y| (|U| from associate() using prior mean)
    {
        const double imgArea =
            static_cast<double>(camera_.imageSize.width) *
            static_cast<double>(camera_.imageSize.height);

        int Ucount = 0;
        const std::size_t nL = sys.numberLandmarks();

        for (std::size_t j = 0; j < nL; ++j)
        {
            // Skip if associated this frame
            if (j < idxFeatures_.size() && idxFeatures_[j] >= 0) continue;
            // Skip if landmark has no tag ID yet
            if (j >= id_by_landmark_.size() || id_by_landmark_[j] < 0) continue;
            // Count if predicted visible at prior mean
            if (j < is_visible_.size() && is_visible_[j]) ++Ucount;
        }

        if (Ucount > 0 && imgArea > 0.0) {
            logL -= 4.0 * static_cast<double>(Ucount) * std::log(imgArea);
        }
    }

    clearAssociationScratch();
    return logL;
}

/*
Log-likelihood + gradient
∇L(x) =  ∑ σ⁻² Jᵀ r, where J = ∂h/∂x (autodiff); |U| term has zero gradient.
*/
double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system,
    Eigen::VectorXd& g) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size());
    g.setZero();

    if (!tl_idxUseLandmarks.empty())
    {
        const std::size_t k = tl_idxUseLandmarks.size();
        const double inv_R = 1.0 / (sigma_ * sigma_);

        for (std::size_t i = 0; i < k; ++i)
        {
            const std::size_t j = tl_idxUseLandmarks[i];
            const int fi = tl_idxUseFeatures[i];

            // y_i
            Eigen::Matrix<double,8,1> yi;
            for (int c = 0; c < 4; ++c)
                yi.segment<2>(2*c) = Y_.col(4*fi + c);

            // Autodiff to get h and J
            using autodiff::dual;
            using autodiff::jacobian;
            using autodiff::wrt;
            using autodiff::at;
            using autodiff::val;

            Eigen::Matrix<dual, Eigen::Dynamic, 1> xdual = x.cast<dual>();
            auto hfun = [&](const Eigen::Matrix<dual, Eigen::Dynamic, 1>& xad)
                        -> Eigen::Matrix<dual,8,1> {
                return predictTagCornersT<dual>(xad, sys, j);
            };

            Eigen::Matrix<dual,8,1> hdual;
            Eigen::MatrixXd Ji = jacobian(hfun, wrt(xdual), at(xdual), hdual);

            Eigen::Matrix<double,8,1> hi;
            for (int t = 0; t < 8; ++t) hi(t) = val(hdual(t));

            const Eigen::Matrix<double,8,1> ri = yi - hi;

            // ∇ contribution
            g.noalias() += inv_R * (Ji.transpose() * ri);
        }
    }

    // |U| term contributes zero gradient (piecewise-constant in x).
    return logLikelihood(x, system);
}

/*
Log-likelihood + gradient + Gauss–Newton Hessian
H ≈ −∑ σ⁻² Jᵀ J ; |U| term contributes zero Hessian.
*/
double MeasurementSLAMUniqueTagBundle::logLikelihood(
    const Eigen::VectorXd& x,
    const SystemEstimator& system,
    Eigen::VectorXd& g,
    Eigen::MatrixXd& H) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size()); g.setZero();
    H.resize(x.size(), x.size()); H.setZero();

    if (!tl_idxUseLandmarks.empty())
    {
        const std::size_t k = tl_idxUseLandmarks.size();
        const double inv_R = 1.0 / (sigma_ * sigma_);

        for (std::size_t i = 0; i < k; ++i)
        {
            const std::size_t j = tl_idxUseLandmarks[i];
            const int fi = tl_idxUseFeatures[i];

            // y_i
            Eigen::Matrix<double,8,1> yi;
            for (int c = 0; c < 4; ++c)
                yi.segment<2>(2*c) = Y_.col(4*fi + c);

            // Autodiff for h and J
            using autodiff::dual;
            using autodiff::jacobian;
            using autodiff::wrt;
            using autodiff::at;
            using autodiff::val;

            Eigen::Matrix<dual, Eigen::Dynamic, 1> xdual = x.cast<dual>();
            auto hfun = [&](const Eigen::Matrix<dual, Eigen::Dynamic, 1>& xad)
                        -> Eigen::Matrix<dual,8,1> {
                return predictTagCornersT<dual>(xad, sys, j);
            };

            Eigen::Matrix<dual,8,1> hdual;
            Eigen::MatrixXd Ji = jacobian(hfun, wrt(xdual), at(xdual), hdual);

            // Gauss–Newton Hessian term
            H.noalias() += -inv_R * (Ji.transpose() * Ji);
        }
    }

    // |U| term contributes zero Hessian.
    return logLikelihood(x, system, g);
}

// Explicit template instantiations for predictTagCornersT
template Eigen::Matrix<double,8,1>
MeasurementSLAMUniqueTagBundle::predictTagCornersT<double>(
    const Eigen::Matrix<double, Eigen::Dynamic, 1>&,
    const SystemSLAM&, std::size_t) const;

template Eigen::Matrix<autodiff::dual,8,1>
MeasurementSLAMUniqueTagBundle::predictTagCornersT<autodiff::dual>(
    const Eigen::Matrix<autodiff::dual, Eigen::Dynamic, 1>&,
    const SystemSLAM&, std::size_t) const;
