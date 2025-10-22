#include <doctest/doctest.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemSLAMPointLandmarks.h"
#include "../../src/Camera.h"
#include "../../src/Pose.hpp"
#include "../../src/rotation.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision,0,", ",";\n","","","[","]")));
#endif

// Local helper: load calibration used by the pipeline.
static Camera loadTestCamera()
{
    Camera cam;
    std::filesystem::path cameraPath("test/data/camera.xml");
    REQUIRE(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    REQUIRE(fs.isOpened());
    fs["camera"] >> cam;
    return cam;
}

// -------------------------------------------------------------------------------------------------
// SystemSLAMPointLandmarks indexing
// Purpose:
//  • testing that numberLandmarks() and landmarkPositionIndex(k) follow 12 + 3k indexing.
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAMPointLandmarks indexing and counts")
{
    GIVEN("A Gaussian density with body + 2 point landmarks (3D each)")
    {
        const int nx = 12 + 2*3;
        auto g = GaussianInfo<double>::fromSqrtMoment(
            Eigen::VectorXd::Zero(nx), Eigen::MatrixXd::Identity(nx,nx)
        );
        SystemSLAMPointLandmarks sys(g);

        THEN("Indices and counts are correct")
        {
            CHECK(sys.numberLandmarks() == 2);
            CHECK(sys.landmarkPositionIndex(0) == 12);
            CHECK(sys.landmarkPositionIndex(1) == 15);
        }
    }
}  // derived from prior indexing checks :contentReference[oaicite:0]{index=0}

// -------------------------------------------------------------------------------------------------
// SystemSLAMPointLandmarks appendFromDuckDetections
// Purpose:
//  • testing that valid detections append one landmark with expected world position.
//  • testing for correct use of fx,fy, pixel ray, and depth from area.
// -------------------------------------------------------------------------------------------------
SCENARIO("SystemSLAMPointLandmarks: appendFromDuckDetections initializes landmarks")
{
    GIVEN("A SystemSLAMPointLandmarks instance and a camera")
    {
        // Base 12-state system at a tilted pose (non-trivial T_nc)
        Eigen::VectorXd mu(12);
        mu.setZero();
        mu.segment<3>(6) << 0.0, 0.0, -1.0;
        mu.segment<3>(9) << -M_PI/2.0, -M_PI/2.0, 0.0;
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(12, 12);
        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
        SystemSLAMPointLandmarks system(p0);
        Camera cam = loadTestCamera();

        WHEN("appendFromDuckDetections is called with one valid detection")
        {
            Eigen::Matrix<double, 2, Eigen::Dynamic> Yuv(2, 1);
            Yuv.col(0) << 960, 720; // near image center
            Eigen::VectorXd A(1);
            A(0) = 1000;
            const double fx = cam.cameraMatrix.at<double>(0, 0);
            const double fy = cam.cameraMatrix.at<double>(1, 1);
            const double duck_r_m = 0.05;
            const double pos_sigma_m = 0.1;

            system.appendFromDuckDetections(cam, Yuv, A, fx, fy, duck_r_m, pos_sigma_m);

            THEN("A landmark is added with expected position from depth & ray")
            {
                REQUIRE(system.numberLandmarks() == 1);

                const double depth = std::sqrt((fx * fy * std::numbers::pi * duck_r_m * duck_r_m) / A(0));
                const cv::Vec3d uPCc_cv = cam.pixelToVector(cv::Vec2d(960, 720));
                Eigen::Vector3d uPCc_eigen;
                cv::cv2eigen(uPCc_cv, uPCc_eigen);
                const Eigen::Vector3d rPCc = depth * uPCc_eigen;

                const Pose<double> Tnb(rpy2rot(mu.segment<3>(9)), mu.segment<3>(6));
                const Pose<double> Tnc = cam.bodyToCamera(Tnb);
                const Eigen::Vector3d expected = Tnc.rotationMatrix * rPCc + Tnc.translationVector;

                auto density = system.landmarkPositionDensity(0);
                CHECK(density.mean()(0) == doctest::Approx(expected(0)));
                CHECK(density.mean()(1) == doctest::Approx(expected(1)));
                CHECK(density.mean()(2) == doctest::Approx(expected(2)));
            }
        }
    }
}  // adapted and merged for clarity :contentReference[oaicite:1]{index=1}
