#include <cstddef>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "Camera.h"
#include "Measurement.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMPointBundle.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <iostream>

namespace {
    // Thread-local scratch for the current evaluation chain
    thread_local bool tl_assoc_ready = false;
    thread_local std::vector<std::size_t> tl_idxUseLandmarks;
    thread_local std::vector<int>         tl_idxUseFeatures;

    inline void ensureAssociatedOnce(const SystemSLAM& sys,
                                     const Eigen::Matrix<double,2,Eigen::Dynamic>& Y,
                                     const std::vector<int>& idxFeatures)
    {
        if (tl_assoc_ready) return; // already built for this chain

        tl_idxUseLandmarks.clear();
        tl_idxUseFeatures.clear();
        tl_idxUseLandmarks.reserve(sys.numberLandmarks());
        tl_idxUseFeatures.reserve(sys.numberLandmarks());

        const std::size_t nL = sys.numberLandmarks();
        for (std::size_t j = 0; j < nL; ++j)
        {
            if (j < idxFeatures.size()) {
                const int fi = idxFeatures[j];
                if (fi >= 0 && fi < Y.cols()) {
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
}


MeasurementPointBundle::MeasurementPointBundle(double time, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_(1.0) // TODO: Assignment(s)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    // updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM * MeasurementPointBundle::clone() const
{
    return new MeasurementPointBundle(*this);
}

Eigen::VectorXd MeasurementPointBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y(Y_.size());
    throw std::runtime_error("Not implemented");
    return y;
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x,
                                             const SystemEstimator & system) const
{
    const SystemSLAM & sys = dynamic_cast<const SystemSLAM &>(system);

    // Build selection once per evaluation chain
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    if (tl_idxUseLandmarks.empty()) {
        clearAssociationScratch();
        return 0.0;
    }

    const std::size_t k = tl_idxUseLandmarks.size();

    // Stack used measurements y = [u1 v1 u2 v2 ...]^T
    Eigen::VectorXd y(2 * k);
    for (std::size_t i = 0; i < k; ++i)
        y.segment<2>(2 * i) = Y_.col(tl_idxUseFeatures[i]);

    // Predict bundle (no Jacobian needed here)
    Eigen::MatrixXd Jdummy;
    Eigen::VectorXd h = predictFeatureBundle(x, Jdummy, sys, tl_idxUseLandmarks);

    const double invR = 1.0 / (sigma_ * sigma_);
    const Eigen::VectorXd r = y - h;

    const double val = -0.5 * invR * r.squaredNorm(); // constants dropped

    // End of the chain: clear scratch so the next evaluation recomputes as needed
    clearAssociationScratch();
    return val;
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x,
                                             const SystemEstimator & system,
                                             Eigen::VectorXd & g) const
{
    const SystemSLAM & sys = dynamic_cast<const SystemSLAM &>(system);

    // Build selection once per evaluation chain (no-op if already done)
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size());
    g.setZero();

    if (!tl_idxUseLandmarks.empty())
    {
        const std::size_t k = tl_idxUseLandmarks.size();

        Eigen::VectorXd y(2 * k);
        for (std::size_t i = 0; i < k; ++i)
            y.segment<2>(2 * i) = Y_.col(tl_idxUseFeatures[i]);

        Eigen::MatrixXd J;                         // (2k x nx)
        Eigen::VectorXd h = predictFeatureBundle(x, J, sys, tl_idxUseLandmarks);

        const double invR = 1.0 / (sigma_ * sigma_);
        const Eigen::VectorXd r = y - h;

        // ∇ logL = invR * J^T r
        g.noalias() = invR * (J.transpose() * r);
    }

    // Build-on style: return via scalar-only overload (will reuse selection and clear it)
    return logLikelihood(x, system);
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x,
                                             const SystemEstimator & system,
                                             Eigen::VectorXd & g,
                                             Eigen::MatrixXd & H) const
{
    const SystemSLAM & sys = dynamic_cast<const SystemSLAM &>(system);

    // Build selection once per evaluation chain (no-op if already done)
    ensureAssociatedOnce(sys, Y_, idxFeatures_);

    g.resize(x.size());               g.setZero();
    H.resize(x.size(), x.size());     H.setZero();

    if (!tl_idxUseLandmarks.empty())
    {
        const std::size_t k = tl_idxUseLandmarks.size();

        Eigen::VectorXd y(2 * k);
        for (std::size_t i = 0; i < k; ++i)
            y.segment<2>(2 * i) = Y_.col(tl_idxUseFeatures[i]);

        Eigen::MatrixXd J;                         // (2k x nx)
        Eigen::VectorXd h = predictFeatureBundle(x, J, sys, tl_idxUseLandmarks);

        const double invR = 1.0 / (sigma_ * sigma_);
        const Eigen::VectorXd r = y - h;

        // ∇ logL and (Gauss–Newton) Hessian of log-likelihood
        g.noalias() = invR * (J.transpose() * r);
        H.noalias() = -invR * (J.transpose() * J);
    }

    // Build-on style: return via the (value,grad) overload (selection stays cached)
    return logLikelihood(x, system, g);
}

void MeasurementPointBundle::update(SystemBase & system)
{
    SystemSLAM & systemSLAM = dynamic_cast<SystemSLAM &>(system);

    // TODO: Assignment(s)
    // Identify landmarks with matching features (data association)
    // Remove failed landmarks from map (consecutive failures to match)
    // Identify surplus features that do not correspond to landmarks in the map
    // Initialise up to Nmax – N new landmarks from best surplus features

    // Select all current landmarks (your Plot and bundle maths expect a fixed ordering)
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);

    // Fill idxFeatures_ by data association:
    //  - By default this calls SNN/compatibility (MeasurementPointBundle::associate)
    //  - For Scenario 1 we constructed MeasurementSLAMUniqueTagBundle, which overrides
    //    associate(...) to do ID-based matching.
    associate(systemSLAM, idxLandmarks); // populates idxFeatures_

    // Optional: here is where you could drop consistently-unmatched landmarks or spawn new ones
    // from surplus features. For Scenario 1 we keep it minimal.
    
    Measurement::update(system);    // Do the actual measurement update
}

// Image feature location for a given landmark and Jacobian
Eigen::Vector2d MeasurementPointBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Set elements of J
    // TODO: Lab 8 (optional)
    //return predictFeature(x, system, idxLandmark);
    // Note: If you use autodiff, return the evaluated function value (cast with double scalar type) instead of calling predictFeature as above

    // iii) Auto diff ONLY ----------------------------------------------------------------
    using autodiff::dual;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;
    using autodiff::val;

    // Promote x to dual
    Eigen::VectorX<dual> xdual = x.cast<dual>();

    // h(x): vector-valued (2x1) functor
    auto h = [&](const Eigen::VectorX<dual>& xad) -> Eigen::Vector2<dual>
    {
        return predictFeature<dual>(xad, system, idxLandmark);
    };

    // J = ∂h/∂x, y = h(x)
    Eigen::Vector2<dual> ydual;
    J = jacobian(h, wrt(xdual), at(xdual), ydual);    // (2 x nx)
    // std::cerr << "dhj/dx =\n" << J << '\n';

    // Return plain double
    Eigen::Vector2d y;
    y << val(ydual(0)), val(ydual(1));
    return y;
    // iii) Auto diff ONLY ----------------------------------------------------------------
}

// Density of image feature location for a given landmark
GaussianInfo<double> MeasurementPointBundle::predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 2;

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::VectorXd ya = predictFeature(x, J, system, idxLandmark) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };
    
    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

// Image feature locations for a bundle of landmarks
Eigen::VectorXd MeasurementPointBundle::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = system.density.dim();
    assert(x.size() == nx);

    Eigen::VectorXd h(2*nL);
    J.resize(2*nL, nx);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::MatrixXd Jfeature;
        Eigen::Vector2d rQOi = predictFeature(x, Jfeature, system, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 9
        h.segment<2>(2*i) = rQOi;
        // Set pair of rows of J
        // TODO: Lab 9
        J.middleRows(2*i, 2) = Jfeature;
    }
    return h;
}

// Density of image features for a set of landmarks
GaussianInfo<double> MeasurementPointBundle::predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 2*idxLandmarks.size();

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::VectorXd ya = predictFeatureBundle(x, J, system, idxLandmarks) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };

    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

#include "association_util.h"
const std::vector<int> & MeasurementPointBundle::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    GaussianInfo<double> featureBundleDensity = predictFeatureBundleDensity(system, idxLandmarks);
    snn(system, featureBundleDensity, idxLandmarks, Y_, camera_, idxFeatures_);
    return idxFeatures_;
}
