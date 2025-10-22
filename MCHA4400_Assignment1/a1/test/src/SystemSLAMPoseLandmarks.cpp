#include <doctest/doctest.h>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemSLAM.h"
#include "../../src/SystemSLAMPoseLandmarks.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision,0,", ",";\n","","","[","]")));
#endif

// Helper: n = 12 + 6·L (pose-landmarks). Order: [v_B, ω_B, r_BN^n, Θ, {landmarks...}].
static GaussianInfo<double> makeDensity(int L)
{
    const int n = 12 + 6*L;
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
    mu.segment<3>(6) << 1,2,3; // r_BN^n (non-zero)
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n);
    return GaussianInfo<double>::fromSqrtMoment(mu, S);
}

static Eigen::VectorXd makeStateFromMu(const GaussianInfo<double>& d) { return d.mean(); }

// -------------------------------------------------------------------------------------------------
// SystemSLAMPoseLandmarks dynamics invariants
// Purpose:
//  • testing that zero body motion ⇒ zero derivatives (base + landmarks).
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAM dynamics invariants at zero body motion")
{
    GIVEN("Theta=0, v=0, w=0 and arbitrary position/landmarks")
    {
        auto density = makeDensity(/*L=*/2);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = makeStateFromMu(density);

        x.segment<3>(0).setZero(); // v_B
        x.segment<3>(3).setZero(); // ω_B
        x.segment<3>(9).setZero(); // Θ

        WHEN("f(x) is evaluated")
        {
            Eigen::VectorXd f = sys.dynamics(0.0, x, Eigen::VectorXd());

            THEN("Base derivatives are zero")
            {
                REQUIRE(f.size() == x.size());
                CHECK(f.segment<3>(0).isZero(0));  // v̇_B
                CHECK(f.segment<3>(3).isZero(0));  // ω̇_B
                CHECK(f.segment<3>(6).isZero(0));  // ṙ_BN^n
                CHECK(f.segment<3>(9).isZero(0));  // Θ̇
            }

            THEN("Landmark derivatives are zero")
            {
                CHECK(f.segment<6>(12 + 0*6).isZero(0));
                CHECK(f.segment<6>(12 + 1*6).isZero(0));
            }
        }
    }
}  // consolidated from dynamics invariants :contentReference[oaicite:2]{index=2}

// -------------------------------------------------------------------------------------------------
// SystemSLAMPoseLandmarks kinematics at Θ=0
// Purpose:
//  • testing that ṙ = v_B and Θ̇ = ω_B at Θ=0 (R=I,T=I).
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAM kinematics at zero attitude (R=I, T=I)")
{
    GIVEN("Theta=0 with nonzero body velocities")
    {
        auto density = makeDensity(/*L=*/1);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = makeStateFromMu(density);

        x.segment<3>(0) << 1.0, 0.5, -0.2;   // v_B
        x.segment<3>(3) << 0.1, -0.2, 0.3;  // ω_B
        x.segment<3>(9).setZero();          // Θ

        WHEN("f(x) is evaluated")
        {
            Eigen::VectorXd f = sys.dynamics(0.0, x, Eigen::VectorXd());

            THEN("ṙ equals v_B")
            {
                CHECK(f.segment<3>(6).isApprox(x.segment<3>(0)));
            }

            THEN("Θ̇ equals ω_B")
            {
                CHECK(f.segment<3>(9).isApprox(x.segment<3>(3)));
            }
        }
    }
}  // retained from prior kinematics test :contentReference[oaicite:3]{index=3}

// -------------------------------------------------------------------------------------------------
// SystemSLAMPoseLandmarks indexing and append
// Purpose:
//  • testing that appendLandmark increases count and uses base index 12 (+6 per landmark).
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAMPoseLandmarks indexing and appendLandmark")
{
    GIVEN("An empty pose-landmark SLAM system")
    {
        const int nx = 12;
        auto g = GaussianInfo<double>::fromSqrtMoment(
            Eigen::VectorXd::Zero(nx), Eigen::MatrixXd::Identity(nx,nx)
        );
        SystemSLAMPoseLandmarks sys(g);

        THEN("appendLandmark increments size and indices as expected")
        {
            Eigen::Vector3d r(1,2,3);
            Eigen::Vector3d th(0.1,0.2,0.3);
            Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Identity();

            std::size_t idx = sys.appendLandmark(r, th, Sj);

            CHECK(idx == 0);
            CHECK(sys.numberLandmarks() == 1);
            CHECK(sys.landmarkPositionIndex(0) == 12);
        }
    }
}  // merged pose-indexing into pose file :contentReference[oaicite:4]{index=4}
