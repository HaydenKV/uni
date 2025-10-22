#include "MeasurementSLAMDuckBundle.h"
#include <cassert>
#include <numeric>
#include "SystemEstimator.h"
#include "SystemBase.h"
#include "Pose.hpp"
#include "association_util.h"

// Constructor
MeasurementSLAMDuckBundle::MeasurementSLAMDuckBundle(double time,
                                                     const Eigen::Matrix<double,2,Eigen::Dynamic>& Yuv,
                                                     const Eigen::VectorXd& A,
                                                     const Camera& camera,
                                                     double sigma_c_px,
                                                     double sigma_a_px2,
                                                     double duck_radius_m)
: MeasurementSLAM(time, camera)
, Yuv_(Yuv)
, A_(A)
, sigma_c_(sigma_c_px)
, sigma_a_(sigma_a_px2)
, duck_r_m_(duck_radius_m)
, fx_(camera.cameraMatrix.at<double>(0,0))
, fy_(camera.cameraMatrix.at<double>(1,1))
{
    assert(Yuv_.cols() == A_.size() && "DuckBundle: Y(2xN) and A(N) must match");
    assert(sigma_c_ > 0.0 && sigma_a_ > 0.0 && duck_r_m_ > 0.0);
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM* MeasurementSLAMDuckBundle::clone() const
{
    auto* m = new MeasurementSLAMDuckBundle(time_, Yuv_, A_, camera_,
                                            sigma_c_, sigma_a_, duck_r_m_);
    m->idxFeatures_ = idxFeatures_;
    m->is_effectively_associated_ = is_effectively_associated_;
    return m;
}

// Prediction function for a single landmark (for doubles only)
Eigen::Vector3d
MeasurementSLAMDuckBundle::predictDuckT(const Eigen::VectorXd& x,
                                        const SystemSLAM& system,
                                        std::size_t idxLandmark) const
{
    Pose<double> Tnc;
    Tnc.translationVector = SystemSLAM::cameraPosition(camera_, x);
    Tnc.rotationMatrix    = SystemSLAM::cameraOrientation(camera_, x);

    const std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3d rLNn = x.segment<3>(idx);

    const Eigen::Matrix3d Rcn = Tnc.rotationMatrix.transpose();
    const Eigen::Vector3d rLCc = Rcn * (rLNn - Tnc.translationVector);

    const Eigen::Vector2d uv = camera_.vectorToPixel(rLCc);

    const double depth = rLCc.norm();
    const double A = (fx_ * fy_ * M_PI * duck_r_m_ * duck_r_m_) / (depth * depth);

    Eigen::Vector3d h;
    h << uv(0), uv(1), A;
    return h;
}

// Predicts a single landmark and computes Jacobian via numerical differentiation
Eigen::Vector3d
MeasurementSLAMDuckBundle::predictDuck(const Eigen::VectorXd& x,
                                       Eigen::MatrixXd& J,
                                       const SystemSLAM& system,
                                       std::size_t idxLandmark) const
{
    const double eps = 1e-6;
    J.resize(3, x.size());
    Eigen::VectorXd x_perturbed = x;

    Eigen::Vector3d h_base = predictDuckT(x, system, idxLandmark);

    for (int i = 0; i < x.size(); ++i) {
        x_perturbed(i) += eps;
        Eigen::Vector3d h_perturbed = predictDuckT(x_perturbed, system, idxLandmark);
        J.col(i) = (h_perturbed - h_base) / eps;
        x_perturbed(i) = x(i);
    }
    return h_base;
}

// Stacks predictions for a bundle of landmarks
Eigen::VectorXd
MeasurementSLAMDuckBundle::predictDuckBundle(const Eigen::VectorXd& x,
                                             Eigen::MatrixXd& J,
                                             const SystemSLAM& system,
                                             const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nL = idxLandmarks.size();
    const std::size_t nx = system.density.dim();

    Eigen::VectorXd h(3 * nL);
    J.resize(3 * nL, nx);

    for (std::size_t i = 0; i < nL; ++i) {
        Eigen::MatrixXd Ji;
        Eigen::Vector3d hi = predictDuck(x, Ji, system, idxLandmarks[i]);
        h.segment<3>(3 * i) = hi;
        J.middleRows(3 * i, 3) = Ji;
    }
    return h;
}

// Centroid-only density for SNN and Plotting
GaussianInfo<double> MeasurementSLAMDuckBundle::predictCentroidBundleDensity(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks) const
{
    const std::size_t nx = system.density.dim();
    const std::size_t L  = idxLandmarks.size();
    const std::size_t ny = 2 * L;

    const auto func = [&](const Eigen::VectorXd& xv, Eigen::MatrixXd& Ja)
    {
        const Eigen::VectorXd x = xv.head(nx);
        const Eigen::VectorXd v = xv.tail(ny);

        Eigen::VectorXd h2D(ny);
        Eigen::MatrixXd J2D(ny, nx);

        for (std::size_t i = 0; i < L; ++i) {
            Eigen::MatrixXd Ji;
            Eigen::Vector3d hi = predictDuck(x, Ji, system, idxLandmarks[i]);
            h2D.segment<2>(2*i) = hi.head<2>();
            J2D.block(2*i, 0, 2, nx) = Ji.topRows(2);
        }

        Ja.resize(ny, nx + ny);
        Ja << J2D, Eigen::MatrixXd::Identity(ny, ny);
        return h2D + v;
    };

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(ny, ny);
    for (std::size_t i = 0; i < L; ++i) {
        S(2*i + 0, 2*i + 0) = sigma_c_;
        S(2*i + 1, 2*i + 1) = sigma_c_;
    }

    auto pv  = GaussianInfo<double>::fromSqrtMoment(S);
    auto pxv = system.density * pv;
    return pxv.affineTransform(func);
}

GaussianInfo<double>
MeasurementSLAMDuckBundle::predictFeatureDensity(const SystemSLAM& system, std::size_t idxLandmark) const
{
    return predictCentroidBundleDensity(system, {idxLandmark});
}

GaussianInfo<double>
MeasurementSLAMDuckBundle::predictFeatureBundleDensity(const SystemSLAM& system, const std::vector<std::size_t>& idxLandmarks) const
{
    return predictCentroidBundleDensity(system, idxLandmarks);
}

const std::vector<int>& MeasurementSLAMDuckBundle::associate(
    const SystemSLAM& system,
    const std::vector<std::size_t>& idxLandmarks)
{
    const std::size_t nL = system.numberLandmarks();
    is_effectively_associated_.assign(nL, false);
    idxFeatures_.assign(nL, -1);

    if (idxLandmarks.empty() || Y().cols() == 0) return idxFeatures_;

    GaussianInfo<double> y2D = predictCentroidBundleDensity(system, idxLandmarks);
    snn(system, y2D, idxLandmarks, Y(), camera_, idxFeatures_);

    for (std::size_t j = 0; j < idxFeatures_.size(); ++j)
        if (idxFeatures_[j] >= 0) is_effectively_associated_[j] = true;

    return idxFeatures_;
}

void MeasurementSLAMDuckBundle::update(SystemBase& systemBase)
{
    SystemSLAM& sys = dynamic_cast<SystemSLAM&>(systemBase);
    std::vector<std::size_t> idxLandmarks(sys.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);

    is_effectively_associated_.assign(sys.numberLandmarks(), false);
    idxFeatures_.assign(sys.numberLandmarks(), -1);

    if (!idxLandmarks.empty() && Y().cols() > 0)
        associate(sys, idxLandmarks);

    Measurement::update(systemBase);
}

// Likelihood functions (unchanged)
static inline void buildInvWeights(std::size_t k, double sc, double sa, Eigen::VectorXd& w)
{
    w.resize(3 * k);
    const double ic = 1.0 / (sc * sc);
    const double ia = 1.0 / (sa * sa);
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t r = 3 * i;
        w(r+0) = ic;
        w(r+1) = ic;
        w(r+2) = ia;
    }
}

Eigen::VectorXd
MeasurementSLAMDuckBundle::simulate(const Eigen::VectorXd& x,
                                    const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    std::vector<std::size_t> idxLandmarks(sys.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
    Eigen::MatrixXd J;
    return predictDuckBundle(x, J, sys, idxLandmarks);
}

double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    std::vector<std::size_t> useL;
    std::vector<int> useF;
    const int N = static_cast<int>(A_.size());
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        const int f = idxFeatures_[j];
        if (f >= 0 && f < N) { useL.push_back(j); useF.push_back(f); }
    }
    const std::size_t k = useL.size();
    if (k == 0) return 0.0;

    Eigen::VectorXd y(3 * k);
    for (std::size_t i = 0; i < k; ++i) {
        y.segment<2>(3*i) = Yuv_.col(useF[i]);
        y(3*i + 2)        = A_(useF[i]);
    }

    Eigen::MatrixXd Jx;
    Eigen::VectorXd h = predictDuckBundle(x, Jx, sys, useL);
    Eigen::VectorXd w;
    buildInvWeights(k, sigma_c_, sigma_a_, w);
    const Eigen::VectorXd r = y - h;
    double ll = 0.0;
    for (int i = 0; i < r.size(); ++i) ll += -0.5 * w(i) * r(i) * r(i);
    return ll;
}

double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system,
                                         Eigen::VectorXd& g) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    std::vector<std::size_t> useL;
    std::vector<int> useF;
    const int N = static_cast<int>(A_.size());
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        const int f = idxFeatures_[j];
        if (f >= 0 && f < N) { useL.push_back(j); useF.push_back(f); }
    }
    const std::size_t k = useL.size();

    g.setZero(x.size());
    if (k > 0) {
        Eigen::VectorXd y(3 * k);
        for (std::size_t i = 0; i < k; ++i) {
            y.segment<2>(3*i) = Yuv_.col(useF[i]);
            y(3*i + 2)        = A_(useF[i]);
        }
        Eigen::MatrixXd J;
        Eigen::VectorXd h = predictDuckBundle(x, J, sys, useL);
        Eigen::VectorXd w;
        buildInvWeights(k, sigma_c_, sigma_a_, w);
        const Eigen::VectorXd r = y - h;
        for (int row = 0; row < J.rows(); ++row) {
            g.noalias() += (w(row) * r(row)) * J.row(row).transpose();
        }
    }
    return logLikelihood(x, system);
}

double
MeasurementSLAMDuckBundle::logLikelihood(const Eigen::VectorXd& x,
                                         const SystemEstimator& system,
                                         Eigen::VectorXd& g,
                                         Eigen::MatrixXd& H) const
{
    const SystemSLAM& sys = dynamic_cast<const SystemSLAM&>(system);
    std::vector<std::size_t> useL;
    std::vector<int> useF;
    const int N = static_cast<int>(A_.size());
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        const int f = idxFeatures_[j];
        if (f >= 0 && f < N) { useL.push_back(j); useF.push_back(f); }
    }
    const std::size_t k = useL.size();

    g.setZero(x.size());
    H = Eigen::MatrixXd::Zero(x.size(), x.size());

    if (k > 0) {
        Eigen::VectorXd y(3 * k);
        for (std::size_t i = 0; i < k; ++i) {
            y.segment<2>(3*i) = Yuv_.col(useF[i]);
            y(3*i + 2)        = A_(useF[i]);
        }
        Eigen::MatrixXd J;
        Eigen::VectorXd h = predictDuckBundle(x, J, sys, useL);
        Eigen::VectorXd w;
        buildInvWeights(k, sigma_c_, sigma_a_, w);
        const Eigen::VectorXd r = y - h;

        Eigen::MatrixXd JW = J;
        for (int row = 0; row < JW.rows(); ++row) JW.row(row) *= std::sqrt(w(row));
        Eigen::VectorXd rW = r;
        for (int i = 0; i < rW.size(); ++i) rW(i) *= std::sqrt(w(i));

        g.noalias() = JW.transpose() * rW;
        H.noalias() = -(JW.transpose() * JW);
    }
    return logLikelihood(x, system, g);
}