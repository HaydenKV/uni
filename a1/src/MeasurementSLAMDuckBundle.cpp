// MeasurementSLAMDuckBundle.cpp
#include "MeasurementSLAMDuckBundle.h"

#include <numeric>
#include <stdexcept>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

// ============================================================================
// Tunables (centralised)
// ============================================================================
// Small positive lower-bound used when converting detection area → depth.
// Prevents division-by-zero and suppresses extreme depths for tiny/degenerate areas.
namespace {
    constexpr double kAreaEps = 1e-6;
}

// ===== Thread-local selection cache (mirrors MeasurementSLAMPointBundle.cpp) =====
// Caches the one-shot association selection within a single evaluation chain so that
// repeated calls (value / grad / Hessian) reuse the same subset consistently.
namespace {
    thread_local bool tl_assoc_ready_duck = false;
    thread_local std::vector<std::size_t> tl_idxUseLandmarks_duck;
    thread_local std::vector<int>         tl_idxUseFeatures_duck;

    // Build association index lists once per evaluation chain.
    inline void ensureDuckAssociatedOnce(const SystemSLAM& sys,
                                         const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                         const std::vector<int>& idxFeatures)
    {
        if (tl_assoc_ready_duck) return;

        tl_idxUseLandmarks_duck.clear();
        tl_idxUseFeatures_duck.clear();
        tl_idxUseLandmarks_duck.reserve(sys.numberLandmarks());
        tl_idxUseFeatures_duck.reserve(sys.numberLandmarks());

        const std::size_t nL = sys.numberLandmarks();
        for (std::size_t j = 0; j < nL; ++j)
        {
            if (j < idxFeatures.size()) {
                const int fi = idxFeatures[j];
                if (fi >= 0 && fi < Y.cols()) {
                    tl_idxUseLandmarks_duck.push_back(j);
                    tl_idxUseFeatures_duck.push_back(fi);
                }
            }
        }
        tl_assoc_ready_duck = true;
    }

    // Reset the thread-local cache after each likelihood evaluation sequence.
    inline void clearDuckAssociationScratch()
    {
        tl_assoc_ready_duck = false;
        tl_idxUseLandmarks_duck.clear();
        tl_idxUseFeatures_duck.clear();
    }
}

// ===== Construction =====
MeasurementSLAMDuckBundle::MeasurementSLAMDuckBundle(double time,
                                                     const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                                                     const Eigen::VectorXd& Avec,
                                                     const Camera& camera,
                                                     double duck_radius_m,
                                                     double sigma_px,
                                                     double sigma_area)
: MeasurementPointBundle(time, Yuv, camera)
, A_(Avec)
, rDuck_(duck_radius_m)
, sigmaA_(sigma_area)
{
    // Reuse base-class pixel noise for (u,v) residuals.
    this->sigma_ = sigma_px;

    // Optional: trust-region update method can be enabled if desired.
    this->updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM* MeasurementSLAMDuckBundle::clone() const
{
    return new MeasurementSLAMDuckBundle(*this);
}

// ===== Single landmark predict: [u, v, A]^T (templated) =====
// Maps world landmark j through the current camera pose into pixel (u,v)
// and predicts apparent area using A = (fx * fy * π * r^2) / Z^2.
template <typename Scalar>
Eigen::Matrix<Scalar,3,1>
MeasurementSLAMDuckBundle::predictDuckFeature(const Eigen::Matrix<Scalar,-1,1>& x,
                                              const SystemSLAM& system,
                                              std::size_t idxLandmark) const
{
    // (1) Camera pose from state.
    Pose<Scalar> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x);
    Tnc.rotationMatrix    = system.cameraOrientation(camera_, x);

    // (2) Landmark world position from state.
    const std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Matrix<Scalar,3,1> rPNn = x.template segment<3>(idx);

    // (3) Transform to camera coordinates: rPCc = Rcn * (rPNn - rCNn).
    const Eigen::Matrix<Scalar,3,3> Rcn = Tnc.rotationMatrix.transpose();
    const Eigen::Matrix<Scalar,3,1> rPCc = Rcn * (rPNn - Tnc.translationVector);

    // (4) Project to pixels [u,v] using calibrated, distorted model.
    const Eigen::Matrix<Scalar,2,1> uv = camera_.vectorToPixel(rPCc);

    // (5) Area model: A_hat = (fx * fy * π * r^2) / Z^2.
    const Scalar fx = static_cast<Scalar>(camera_.cameraMatrix.at<double>(0,0));
    const Scalar fy = static_cast<Scalar>(camera_.cameraMatrix.at<double>(1,1));
    const Scalar Zc = rPCc(2);
    const Scalar k  = fx * fy * static_cast<Scalar>(M_PI) * static_cast<Scalar>(rDuck_ * rDuck_);

    Eigen::Matrix<Scalar,3,1> y;
    y.template head<2>() = uv;
    y(2) = k / (Zc * Zc);

    return y;
}

// ===== Single landmark predict with Jacobian (autodiff) =====
Eigen::Vector3d
MeasurementSLAMDuckBundle::predictDuckFeature(const Eigen::VectorXd& x,
                                              Eigen::MatrixXd& J,
                                              const SystemSLAM& system,
                                              std::size_t idxLandmark) const
{
    using autodiff::dual;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;
    using autodiff::val;

    Eigen::Matrix<dual,-1,1> xdual = x.cast<dual>();

    auto h = [&](const Eigen::Matrix<dual,-1,1>& xad) -> Eigen::Matrix<dual,3,1>
    {
        return predictDuckFeature<dual>(xad, system, idxLandmark);
    };

    Eigen::Matrix<dual,3,1> ydual;
    J = jacobian(h, wrt(xdual), at(xdual), ydual);  // (3 x nx)

    Eigen::Vector3d y;
    y << val(ydual(0)), val(ydual(1)), val(ydual(2));
    return y;
}

// ===== Bundle prediction (templated) =====
// Stacks [u v A]^T for a subset of landmarks into a single vector.
template <typename Scalar>
Eigen::Matrix<Scalar,-1,1>
MeasurementSLAMDuckBundle::predictDuckBundle(const Eigen::Matrix<Scalar,-1,1>& x,
                                             const SystemSLAM& system,
                                             const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nL = idxLandmarks.size();
    Eigen::Matrix<Scalar,-1,1> h(3*nL);
    for (std::size_t i = 0; i < nL; ++i)
    {
        auto yi = predictDuckFeature<Scalar>(x, system, idxLandmarks[i]);
        h.template segment<3>(3*i) = yi;
    }
    return h;
}

Eigen::VectorXd
MeasurementSLAMDuckBundle::predictDuckBundle(const Eigen::VectorXd& x,
                                             Eigen::MatrixXd& J,
                                             const SystemSLAM& system,
                                             const std::vector<std::size_t>& idxLandmarks) const
{
    using autodiff::dual;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;
    using autodiff::val;

    Eigen::Matrix<dual,-1,1> xdual = x.cast<dual>();
    const std::size_t nL = idxLandmarks.size();

    auto h = [&](const Eigen::Matrix<dual,-1,1>& xad) -> Eigen::Matrix<dual,-1,1>
    {
        return predictDuckBundle<dual>(xad, system, idxLandmarks);
    };

    Eigen::Matrix<dual,-1,1> ydual(3*nL);
    J = jacobian(h, wrt(xdual), at(xdual), ydual);  // (3nL x nx)

    Eigen::VectorXd y(3*nL);
    for (std::size_t i = 0; i < 3*nL; ++i) y(i) = val(ydual(i));
    return y;
}

// Provide a specific subset of landmarks to consider during association/update.
void MeasurementSLAMDuckBundle::setCandidateLandmarks(const std::vector<std::size_t>& idx)
{
    candidateLandmarks_ = idx;
}

// ===== Log-likelihood (value) =====
// Builds the selection once (thread-local), then evaluates the diagonal-weighted
// Gaussian log-likelihood for stacked residuals (u, v, A) per associated LM.
double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system_) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system_);

    // Build association selection once per evaluation chain.
    ensureDuckAssociatedOnce(sys, Y_, idxFeatures_);

    if (tl_idxUseLandmarks_duck.empty()) {
        clearDuckAssociationScratch();
        return 0.0;
    }

    const std::size_t k = tl_idxUseLandmarks_duck.size();

    // Assemble measurement vector y = [u v A u v A ...]^T
    Eigen::VectorXd y(3*k);
    for (std::size_t i = 0; i < k; ++i) {
        const int fi = tl_idxUseFeatures_duck[i];
        y.segment<2>(3*i)   = Y_.col(fi);
        y(3*i + 2)          = A_(fi);
    }

    // Predict & residual
    Eigen::MatrixXd Jdummy;
    Eigen::VectorXd h = predictDuckBundle(x, Jdummy, sys, tl_idxUseLandmarks_duck);
    const Eigen::VectorXd r = y - h;

    // Diagonal inverse covariance: invR = diag(1/σ_px^2, 1/σ_px^2, 1/σ_A^2, ...)
    const double invRuv = 1.0 / (sigma_ * sigma_);
    const double invRA  = 1.0 / (sigmaA_ * sigmaA_);

    double val = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        const double ru = r(3*i + 0);
        const double rv = r(3*i + 1);
        const double rA = r(3*i + 2);
        val += -0.5 * (invRuv * (ru*ru + rv*rv) + invRA * (rA*rA));
    }

    clearDuckAssociationScratch();
    return val;
}

// ===== Log-likelihood (value + gradient) =====
// Uses cached association; computes y, h(x), residuals, and accumulates J^T W r.
double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system_,
                                         Eigen::VectorXd& g) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system_);

    ensureDuckAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size());
    g.setZero();

    if (!tl_idxUseLandmarks_duck.empty())
    {
        const std::size_t k = tl_idxUseLandmarks_duck.size();

        Eigen::VectorXd y(3*k);
        for (std::size_t i = 0; i < k; ++i) {
            const int fi = tl_idxUseFeatures_duck[i];
            y.segment<2>(3*i)   = Y_.col(fi);
            y(3*i + 2)          = A_(fi);
        }

        Eigen::MatrixXd J;                  // (3k x nx)
        Eigen::VectorXd h = predictDuckBundle(x, J, sys, tl_idxUseLandmarks_duck);
        const Eigen::VectorXd r = y - h;

        const double invRuv = 1.0 / (sigma_ * sigma_);
        const double invRA  = 1.0 / (sigmaA_ * sigmaA_);

        // g = J^T * W * r  (W is diagonal).
        for (std::size_t i = 0; i < k; ++i) {
            const double w0 = invRuv, w1 = invRuv, w2 = invRA;
            g.noalias() += J.row(3*i + 0).transpose() * (w0 * r(3*i + 0));
            g.noalias() += J.row(3*i + 1).transpose() * (w1 * r(3*i + 1));
            g.noalias() += J.row(3*i + 2).transpose() * (w2 * r(3*i + 2));
        }
    }

    // Also return scalar value (reusing the selection and clearing it).
    return logLikelihood(x, system_);
}

// ===== Log-likelihood (value + gradient + Hessian) =====
// Computes Gauss–Newton approximation: H ≈ -J^T W J (block-diagonal per [u v A]),
// together with the gradient accumulation.
double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system_,
                                         Eigen::VectorXd& g,
                                         Eigen::MatrixXd& H) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system_);

    ensureDuckAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size()); g.setZero();
    H.resize(x.size(), x.size()); H.setZero();

    if (!tl_idxUseLandmarks_duck.empty())
    {
        const std::size_t k = tl_idxUseLandmarks_duck.size();

        Eigen::VectorXd y(3*k);
        for (std::size_t i = 0; i < k; ++i) {
            const int fi = tl_idxUseFeatures_duck[i];
            y.segment<2>(3*i)   = Y_.col(fi);
            y(3*i + 2)          = A_(fi);
        }

        Eigen::MatrixXd J;                  // (3k x nx)
        Eigen::VectorXd h = predictDuckBundle(x, J, sys, tl_idxUseLandmarks_duck);
        const Eigen::VectorXd r = y - h;

        const double invRuv = 1.0 / (sigma_ * sigma_);
        const double invRA  = 1.0 / (sigmaA_ * sigmaA_);

        // g = J^T * W * r,   H ≈ - J^T * W * J.
        for (std::size_t i = 0; i < k; ++i) {
            const double w0 = invRuv, w1 = invRuv, w2 = invRA;
            const auto J0 = J.row(3*i + 0);
            const auto J1 = J.row(3*i + 1);
            const auto J2 = J.row(3*i + 2);

            g.noalias() += J0.transpose() * (w0 * r(3*i + 0));
            g.noalias() += J1.transpose() * (w1 * r(3*i + 1));
            g.noalias() += J2.transpose() * (w2 * r(3*i + 2));

            H.noalias() -= (w0 * (J0.transpose() * J0));
            H.noalias() -= (w1 * (J1.transpose() * J1));
            H.noalias() -= (w2 * (J2.transpose() * J2));
        }
    }

    // Return scalar value (gradient overload keeps selection cached).
    return logLikelihood(x, system_, g);
}

// ===== Update: perform association on 2D and then call base Measurement::update =====
// Associates features to candidate landmarks using the base 2D SNN/compatibility
// logic (fills idxFeatures_), then executes the standard nonlinear update that
// uses the (u,v,A) likelihood defined above.
void
MeasurementSLAMDuckBundle::update(SystemBase& system)
{
    SystemSLAM& sys = dynamic_cast<SystemSLAM&>(system);

    // Use caller-provided candidate landmarks if any; otherwise use all.
    std::vector<std::size_t> idxLandmarks;
    if (!candidateLandmarks_.empty()) {
        idxLandmarks = candidateLandmarks_;
    } else {
        idxLandmarks.resize(sys.numberLandmarks());
        std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
    }

    // Use SNN association from the base (2D) to fill idxFeatures_ (as in labs).
    associate(sys, idxLandmarks);

    // Proceed with the standard nonlinear update using our 3D (u,v,A) likelihood.
    Measurement::update(system);

    // One-shot: clear candidates so next call defaults to "all" unless set again.
    candidateLandmarks_.clear();
}
