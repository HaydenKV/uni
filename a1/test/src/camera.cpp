#include <doctest/doctest.h>
#include <filesystem>
#include <limits>
#include <array>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include "../../src/Pose.hpp"
#include "../../src/Camera.h"

// Camera model checks (assignment: pinhole projection, no distortion in test XML):
//  • FOV: +Z in, −Z / lateral out.
//  • pixel↔ray inversion: vectorToPixel(pixelToVector(p)) ≈ p.
//  • world projection: worldToPixel = vectorToPixel ∘ worldToVector.
//  • conservative FOV near right edge (association/gating).
SCENARIO("Camera model")
{
    GIVEN("A camera with no lens distortion")
    {
        std::filesystem::path cameraPath("test/data/camera.xml");
        REQUIRE(std::filesystem::exists(cameraPath));
        REQUIRE(std::filesystem::is_regular_file(cameraPath));

        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        REQUIRE(fs.isOpened());

        Camera cam;
        fs["camera"] >> cam;

        REQUIRE(cam.cameraMatrix.cols == 3);
        REQUIRE(cam.cameraMatrix.rows == 3);
        REQUIRE(cam.cameraMatrix.type() == CV_64F);
        REQUIRE(cam.distCoeffs.cols == 1);
        REQUIRE(cam.distCoeffs.type() == CV_64F);

        // FOV: +Z inside. vectorToPixel(+Z) → (cx,cy).  (assignment: camera projection)
        GIVEN("The positive optical axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 0.0, 1.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));
            CHECK(cam.isVectorWithinFOV(uPCc));

            cv::Vec2d rQOi = cam.vectorToPixel(uPCc);
            const double & cx = cam.cameraMatrix.at<double>(0, 2);
            const double & cy = cam.cameraMatrix.at<double>(1, 2);
            CHECK(rQOi(0) == doctest::Approx(cx));
            CHECK(rQOi(1) == doctest::Approx(cy));
        }

        // FOV: −Z and lateral directions outside (pinhole forward hemisphere only).
        GIVEN("The negative optical axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 0.0, -1.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));
            CHECK_FALSE(cam.isVectorWithinFOV(uPCc));
        }
        GIVEN("The positive horizontal image axis unit vector")
        {
            cv::Vec3d uPCc(1.0, 0.0, 0.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));
            CHECK_FALSE(cam.isVectorWithinFOV(uPCc));
        }
        GIVEN("The positive vertical image axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 1.0, 0.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));
            CHECK_FALSE(cam.isVectorWithinFOV(uPCc));
        }

        // pixel↔ray inversion: vectorToPixel(pixelToVector(p)) ≈ p.  (assignment: projection/inverse)
        GIVEN("A pixel location")
        {
            cv::Vec2d p(5.3, 8.7);
            cv::Vec3d u = cam.pixelToVector(p);
            REQUIRE(cv::norm(u) == doctest::Approx(1.0));
            cv::Vec2d p2 = cam.vectorToPixel(u);
            CHECK(p2(0) == doctest::Approx(p(0)));
            CHECK(p2(1) == doctest::Approx(p(1)));
        }

        // worldToPixel = vectorToPixel ∘ worldToVector.  (assignment: composition)
        GIVEN("An arbitrary world point and body pose")
        {
            cv::Vec3d rPNn(0.141886338627215, 0.421761282626275, 0.915735525189067);

            Pose<double> Tnb;
            Tnb.translationVector << 0.792207329559554, 0.959492426392903, 0.655740699156587;
            Tnb.rotationMatrix <<
                 0.988707451899469, -0.0261040852852265, 0.147567446579119,
                -0.0407096903396656,   0.900896143585899, 0.432121348216568,
                -0.144223076069359,  -0.433249022161017,  0.88966004132231;

            REQUIRE((Tnb.rotationMatrix.transpose()*Tnb.rotationMatrix - Eigen::Matrix3d::Identity()).lpNorm<Eigen::Infinity>() < 100*std::numeric_limits<double>::epsilon());

            cv::Vec2d p1 = cam.worldToPixel(rPNn, Tnb);
            cv::Vec2d p2 = cam.vectorToPixel(cam.worldToVector(rPNn, Tnb));

            CHECK(p1(0) == doctest::Approx(p2(0)));
            CHECK(p1(1) == doctest::Approx(p2(1)));

            // Regression oracle (from provided numbers).
            cv::Vec2d p_oracle(34.039103190885967, 36.073879427476669);
            CHECK(p1(0) == doctest::Approx(p_oracle(0)));
            CHECK(p1(1) == doctest::Approx(p_oracle(1)));
        }

        // Corner gating (assignment: robust visibility/gating): in-bounds accepted; off-image rejected.
        GIVEN("Corner sets clearly inside vs clearly outside the image + margin")
        {
            std::array<cv::Point2f,4> inside{
                cv::Point2f(50.f, 50.f),
                cv::Point2f(400.f, 50.f),
                cv::Point2f(400.f, 400.f),
                cv::Point2f(50.f, 400.f)
            };
            std::array<cv::Point2f,4> outside{
                cv::Point2f(-10.f, -10.f),
                cv::Point2f(100.f, -10.f),
                cv::Point2f(100.f,  10.f),
                cv::Point2f(-10.f,  10.f)
            };

            CHECK(cam.areCornersInside(inside));
            CHECK_FALSE(cam.areCornersInside(outside));
        }

        // Conservative FOV near right edge (assignment: association/gating at borders).
        GIVEN("Rays near the horizontal FOV boundary (conservative gating)")
        {
            const double W = cam.imageSize.width;
            const double H = cam.imageSize.height;

            cv::Vec3d v_in  = cam.pixelToVector(cv::Vec2d(0.90*W, H * 0.5));
            cv::Vec3d v_out = cam.pixelToVector(cv::Vec2d(W + 1.0, H * 0.5));

            CHECK(cam.isVectorWithinFOVConservative(v_in));
            CHECK_FALSE(cam.isVectorWithinFOVConservative(v_out));
        }
    }
}
