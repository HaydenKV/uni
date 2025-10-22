// test/src/camera_vectorToPixel_jacobians.cpp
#include <doctest/doctest.h>
#include <filesystem>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/persistence.hpp>

#include "../../src/Camera.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

using autodiff::dual;
using autodiff::jacobian;
using autodiff::wrt;
using autodiff::at;
using autodiff::val;

// AutoDiff oracle on the templated path
static std::pair<Eigen::Vector2d, Eigen::Matrix<double,2,3>>
autodiff_oracle(const Camera& cam, const Eigen::Vector3d& rPCc)
{
    Eigen::VectorX<dual> x(3);
    x << rPCc(0), rPCc(1), rPCc(2);

    auto f = [&](const Eigen::VectorX<dual>& v)->Eigen::Vector2<dual> {
        Eigen::Vector3<dual> r(v(0), v(1), v(2));
        return cam.vectorToPixel<dual>(r);
    };

    Eigen::Vector2<dual> ydual;
    Eigen::Matrix<double,2,3> J = jacobian(f, wrt(x), at(x), ydual);
    Eigen::Vector2d y(val(ydual(0)), val(ydual(1)));
    return { y, J };
}

// Central FD using the templated *double* path with a relative step
static Eigen::Matrix<double,2,3>
fd_jacobian_double(const Camera& cam, const Eigen::Vector3d& rPCc, double base = 1e-6)
{
    auto f = [&](const Eigen::Vector3d& r)->Eigen::Vector2d {
        return cam.vectorToPixel<double>(r);
    };

    Eigen::Matrix<double,2,3> J;
    for (int j = 0; j < 3; ++j) {
        double h = std::max(1.0, std::abs(rPCc(j))) * base;
        Eigen::Vector3d rp = rPCc; rp(j) += h;
        Eigen::Vector3d rm = rPCc; rm(j) -= h;
        J.col(j) = (f(rp) - f(rm)) / (2.0 * h);
    }
    return J;
}

SCENARIO("Task 2: Camera::vectorToPixel Jacobian (AutoDiff oracle vs finite-difference on double path)")
{
    GIVEN("A camera loaded from test/data/camera.xml")
    {
        std::filesystem::path cameraPath("test/data/camera.xml");
        REQUIRE(std::filesystem::exists(cameraPath));
        REQUIRE(std::filesystem::is_regular_file(cameraPath));

        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        REQUIRE(fs.isOpened());

        Camera cam;
        fs["camera"] >> cam;

        std::vector<Eigen::Vector3d> rays = {
            { 0.10,  0.20, 1.00},
            {-0.05,  0.15, 0.90},
            { 0.50, -0.30, 2.00},
            { 0.00,  0.00, 1.00},
            { 0.20, -0.10, 0.50}
        };

        // Tolerances
        const double tolF = 1e-10;   // function value
        const double tolJ = 1e-6;    // Jacobian
        const double fdBase = 1e-6;  // FD step base

        for (const auto& rPCc : rays)
        {
            WHEN("Comparing AutoDiff oracle vs finite-difference (double path)")
            {
                CAPTURE(rPCc.transpose());

                // AutoDiff oracle
                auto [y_ad, J_ad] = autodiff_oracle(cam, rPCc);

                // FD on templated double path
                Eigen::Matrix<double,2,3> J_fd = fd_jacobian_double(cam, rPCc, fdBase);

                // Function value via templated double
                Eigen::Vector2d y_tpl = cam.vectorToPixel<double>(rPCc);

                THEN("Function values match (AutoDiff vs templated double)")
                {
                    CHECK(y_tpl(0) == doctest::Approx(y_ad(0)).epsilon(tolF).scale(1.0));
                    CHECK(y_tpl(1) == doctest::Approx(y_ad(1)).epsilon(tolF).scale(1.0));
                }

                THEN("AutoDiff Jacobian matches finite-difference Jacobian (double path)")
                {
                    for (int r = 0; r < 2; ++r)
                        for (int c = 0; c < 3; ++c)
                            CHECK(J_ad(r,c) == doctest::Approx(J_fd(r,c)).epsilon(tolJ).scale(1.0));
                }

                // Sanity check vs OpenCV (float) path — loose tolerance
                cv::Vec2d y_cv = cam.vectorToPixel(cv::Vec3d(rPCc(0), rPCc(1), rPCc(2)));
                Eigen::Vector2d y_cv_eig(y_cv[0], y_cv[1]);
                const double tol_cv = 1e-3;
                THEN("Templated double and OpenCV (float) path are close")
                {
                    CHECK(y_tpl(0) == doctest::Approx(y_cv_eig(0)).epsilon(tol_cv).scale(1.0));
                    CHECK(y_tpl(1) == doctest::Approx(y_cv_eig(1)).epsilon(tol_cv).scale(1.0));
                }
            }
        }
    }
}
