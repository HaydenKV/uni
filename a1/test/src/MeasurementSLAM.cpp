#include <doctest/doctest.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <vector>
#include <numbers>

#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemSLAM.h"
#include "../../src/SystemSLAMPoseLandmarks.h"
#include "../../src/SystemSLAMPointLandmarks.h"
#include "../../src/MeasurementSLAMUniqueTagBundle.h"
#include "../../src/MeasurementSLAMDuckBundle.h"
#include "../../src/Camera.h"
#include "../../src/Pose.hpp"
#include "../../src/rotation.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision,0,", ",";\n","","","[","]")));
#endif

// Camera helper
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

// ---- Small helpers for duck tests ----
static SystemSLAMPointLandmarks makeDuckSLAMSystem(const Eigen::Vector3d& landmark_pos_n)
{
    const int n = 12 + 3; // body + 1 point landmark
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
    mu.segment<3>(12) = landmark_pos_n;
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n,n);
    auto density = GaussianInfo<double>::fromSqrtMoment(mu, S);
    return SystemSLAMPointLandmarks(density);
}

// -------------------------------------------------------------------------------------------------
// UniqueTagBundle (1 tag)
// Purpose:
//  • testing that logLikelihood returns correct sizes and first-order identity holds.
// -------------------------------------------------------------------------------------------------
SCENARIO("MeasurementSLAMUniqueTagBundle: log-likelihood API and dimensions")
{
    GIVEN("A system with one tag and a camera")
    {
        Camera cam = loadTestCamera();

        // n=12+6
        const int n = 12 + 6;
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        mu.segment<3>(12) << 0.2, -0.1, 5.0; // tag position
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n,n);
        auto density = GaussianInfo<double>::fromSqrtMoment(mu, S);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = density.mean();

        // Place synthetic corners around the pixel center of the projected tag
        Posed Tnb;
        cv::Vec2d c = cam.worldToPixel(cv::Vec3d(0.2,-0.1,5.0), Tnb);
        auto corners = [&](const cv::Vec2d& ctr){
            const double h = 4.0;
            Eigen::Matrix<double,2,4> Y;
            Y.col(0) << ctr[0]-h, ctr[1]-h;
            Y.col(1) << ctr[0]+h, ctr[1]-h;
            Y.col(2) << ctr[0]+h, ctr[1]+h;
            Y.col(3) << ctr[0]-h, ctr[1]+h;
            return Y;
        };
        Eigen::Matrix<double,2,Eigen::Dynamic> Y(2,4);
        Y.leftCols<4>() = corners(c);
        std::vector<int> ids = {0};

        MeasurementSLAMUniqueTagBundle meas(0.0, Y, cam, ids);

        WHEN("logLikelihood(x,•) is evaluated")
        {
            Eigen::VectorXd g; Eigen::MatrixXd H;
            double l = meas.logLikelihood(x, sys, g, H);

            THEN("Gradient/Hessian sizes match and directional derivative holds")
            {
                CHECK(std::isfinite(l));
                REQUIRE((int)g.size() == n);
                REQUIRE(H.rows() == n);
                REQUIRE(H.cols() == n);

                Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
                d(6) = 1e-3;  d(7) = -1e-3;
                const double eps = 1e-2;
                const double fd  = (meas.logLikelihood(x + eps*d, sys) - l)/eps;
                const double dot = g.dot(d);
                CHECK(fd == doctest::Approx(dot).epsilon(1e-2));
            }
        }
    }
}  // merged from single-tag test :contentReference[oaicite:10]{index=10}

// -------------------------------------------------------------------------------------------------
// UniqueTagBundle (2 tags)
// Purpose:
//  • testing that multi-tag packing and gradient consistency hold with first-order check.
// -------------------------------------------------------------------------------------------------
SCENARIO("MeasurementSLAMUniqueTagBundle: two tags — first-order consistency")
{
    GIVEN("Two pose-landmarks visible and a calibrated camera")
    {
        Camera cam = loadTestCamera();

        const double fx = cam.cameraMatrix.at<double>(0,0);
        const double fy = cam.cameraMatrix.at<double>(1,1);
        const double cx = cam.cameraMatrix.at<double>(0,2);
        const double cy = cam.cameraMatrix.at<double>(1,2);
        const double W  = cam.imageSize.width;
        const double H  = cam.imageSize.height;

        auto pixelToWorldAtDepth = [&](double u, double v, double Z){
            return Eigen::Vector3d((u - cx)/fx * Z, (v - cy)/fy * Z, Z);
        };

        cv::Vec2d c0_px(0.55*W, 0.55*H), c1_px(0.70*W, 0.40*H);
        Eigen::Vector3d tag0_n = pixelToWorldAtDepth(c0_px[0], c0_px[1], 5.0);
        Eigen::Vector3d tag1_n = pixelToWorldAtDepth(c1_px[0], c1_px[1], 6.0);

        const int n = 12 + 2*6;
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        mu.segment<3>(12) = tag0_n;
        mu.segment<3>(18) = tag1_n;
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n,n);
        auto density = GaussianInfo<double>::fromSqrtMoment(mu, S);
        SystemSLAMPoseLandmarks sys(density);
        Eigen::VectorXd x = density.mean();

        auto corners = [&](const cv::Vec2d& ctr){
            const double h = 8.0;
            Eigen::Matrix<double,2,4> Y;
            Y.col(0) << ctr[0]-h, ctr[1]-h;
            Y.col(1) << ctr[0]+h, ctr[1]-h;
            Y.col(2) << ctr[0]+h, ctr[1]+h;
            Y.col(3) << ctr[0]-h, ctr[1]+h;
            return Y;
        };

        Eigen::Matrix<double,2,8> Y;
        Y.leftCols<4>()  = corners(c0_px);
        Y.rightCols<4>() = corners(c1_px);

        std::vector<int> ids = {0, 1};
        MeasurementSLAMUniqueTagBundle meas(0.0, Y, cam, ids);
        meas.setIdByLandmark({0, 1});

        WHEN("Comparing dℓ/dx vs finite differences along d")
        {
            Eigen::VectorXd g; Eigen::MatrixXd H;
            double l0 = meas.logLikelihood(x, sys, g, H);

            Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
            d.segment<3>(6) <<  0.7, -0.3, 0.5;
            d.segment<3>(9) << -0.2,  0.4, 0.1;
            d.normalize();

            const double eps = 1e-3;
            const double fd   = (meas.logLikelihood(x + eps*d, sys) - l0)/eps;
            const double dotg = g.dot(d);
            CHECK(fd == doctest::Approx(dotg).epsilon(5e-2));
        }
    }
}  // merged multi-tag gradient check :contentReference[oaicite:11]{index=11}

// -------------------------------------------------------------------------------------------------
// DuckBundle: predictDuckFeature
// Purpose:
//  • testing that y=[u,v,A] matches pinhole center and A=(fx*fy*π*r²)/Z² on-axis.
//  • testing for Jacobian consistency with finite differences.
// -------------------------------------------------------------------------------------------------
SCENARIO("MeasurementSLAMDuckBundle: predictDuckFeature correctness")
{
    GIVEN("One landmark on the optical axis")
    {
        Camera cam = loadTestCamera();
        const double Z = 2.0;
        auto sys = makeDuckSLAMSystem(Eigen::Vector3d(0,0,Z));
        Eigen::VectorXd x = sys.density.mean();

        MeasurementSLAMDuckBundle meas(0.0, {}, {}, cam, /*r*/0.054);

        WHEN("predictDuckFeature is called")
        {
            Eigen::MatrixXd J;
            Eigen::Vector3d y = meas.predictDuckFeature(x, J, sys, 0);

            THEN("u,v equal principal point and area matches model")
            {
                const double cx = cam.cameraMatrix.at<double>(0,2);
                const double cy = cam.cameraMatrix.at<double>(1,2);
                CHECK(y(0) == doctest::Approx(cx));
                CHECK(y(1) == doctest::Approx(cy));

                const double fx = cam.cameraMatrix.at<double>(0,0);
                const double fy = cam.cameraMatrix.at<double>(1,1);
                const double A  = (fx*fy*std::numbers::pi*0.054*0.054)/(Z*Z);
                CHECK(y(2) == doctest::Approx(A));
            }

            THEN("Jacobian matches finite differences")
            {
                Eigen::VectorXd d = Eigen::VectorXd::Zero(x.size());
                d(12) = 1.0;  // landmark x
                d(7)  = -1.0; // body y
                d.normalize();

                const double eps = 1e-6;
                Eigen::Vector3d y_plus = meas.predictDuckFeature(x + eps*d, J, sys, 0);
                Eigen::Vector3d fd = (y_plus - y)/eps;
                Eigen::Vector3d Jd = J*d;
                CAPTURE_EIGEN(fd);
                CAPTURE_EIGEN(Jd);
                CHECK(fd.isApprox(Jd, 1e-6));
            }
        }
    }
}  // consolidated from duck feature test :contentReference[oaicite:12]{index=12}

// -------------------------------------------------------------------------------------------------
// DuckBundle: logLikelihood
// Purpose:
//  • testing that gradient/Hessian sizes are correct and first-order identity holds.
//  • testing for Gauss-Newton structure H ≈ −JᵀWJ.
// -------------------------------------------------------------------------------------------------
SCENARIO("MeasurementSLAMDuckBundle: logLikelihood API and derivatives")
{
    GIVEN("A camera, system, and one associated duck measurement")
    {
        Camera cam = loadTestCamera();
        auto sys = makeDuckSLAMSystem(Eigen::Vector3d(0.1, -0.2, 3.0));
        Eigen::VectorXd x = sys.density.mean();

        MeasurementSLAMDuckBundle meas_for_pred(0.0, {}, {}, cam, 0.054);
        Eigen::MatrixXd Jp;
        Eigen::Vector3d h_true = meas_for_pred.predictDuckFeature(x, Jp, sys, 0);

        Eigen::Matrix<double,2,1> Yuv;
        Yuv << h_true(0) + 1.0, h_true(1) - 1.0;
        Eigen::Matrix<double,1,1> Avec;
        Avec << h_true(2) + 5.0;

        MeasurementSLAMDuckBundle meas(1.0, Yuv, Avec, cam, 0.054, /*σpx*/2.5, /*σA*/150.0);

        std::vector<std::size_t> idx = {0};
        meas.associate(sys, idx);

        WHEN("Evaluating l(x), g, H")
        {
            Eigen::VectorXd g; Eigen::MatrixXd H;
            double l = meas.logLikelihood(x, sys, g, H);

            THEN("Dimensions are correct and value matches quadratic form")
            {
                CHECK(std::isfinite(l));
                REQUIRE(g.size() == x.size());
                REQUIRE(H.rows() == x.size());
                REQUIRE(H.cols() == x.size());

                Eigen::Vector3d y_meas(Yuv(0,0), Yuv(1,0), Avec(0));
                Eigen::Vector3d r = y_meas - h_true;
                const double inv_var_px = 1.0/(2.5*2.5);
                const double inv_var_A  = 1.0/(150.0*150.0);
                const double expected_l = -0.5*(inv_var_px*(r(0)*r(0)+r(1)*r(1)) + inv_var_A*r(2)*r(2));

                meas.associate(sys, idx); // reset cache for independent call
                CHECK(meas.logLikelihood(x, sys) == doctest::Approx(expected_l));
            }

            THEN("Directional derivative matches gᵀd")
            {
                Eigen::VectorXd d = Eigen::VectorXd::Zero(x.size());
                d(6) = 0.1; d(10) = -0.05; d(14) = 0.2;
                const double eps = 1e-4;

                meas.associate(sys, idx);
                const double l_plus = meas.logLikelihood(x + eps*d, sys);

                const double fd = (l_plus - l)/eps;
                const double dot = g.dot(d);
                CAPTURE(fd); CAPTURE(dot);
                CHECK(fd == doctest::Approx(dot).epsilon(1e-4));
            }

            THEN("Gauss-Newton Hessian matches −JᵀWJ")
            {
                Eigen::MatrixXd J;
                meas.predictDuckFeature(x, J, sys, 0);

                Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
                W(0,0) = 1.0/(2.5*2.5);
                W(1,1) = 1.0/(2.5*2.5);
                W(2,2) = 1.0/(150.0*150.0);

                Eigen::MatrixXd Hexp = -J.transpose()*W*J;
                CAPTURE_EIGEN(H);
                CAPTURE_EIGEN(Hexp);
                CHECK(H.isApprox(Hexp));
            }
        }
    }
}  // consolidated duck likelihood checks :contentReference[oaicite:13]{index=13}
