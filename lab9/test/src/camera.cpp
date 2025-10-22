#include <doctest/doctest.h>
#include <filesystem>
#include <limits>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include "../../src/Pose.hpp"
#include "../../src/Camera.h"

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

        GIVEN("The positive optical axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 0.0, 1.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

            WHEN("Checking field of view")
            {
                bool isWithinFOV = cam.isVectorWithinFOV(uPCc);
                THEN("Vector is within field of view")
                {
                    CHECK(isWithinFOV);
                }
            }

            WHEN("Calling vectorToPixel")
            {
                cv::Vec2d rQOi = cam.vectorToPixel(uPCc);

                THEN("Vector maps to centre of image")
                {
                    const double & cx = cam.cameraMatrix.at<double>(0, 2);
                    const double & cy = cam.cameraMatrix.at<double>(1, 2);

                    CHECK(rQOi(0) == doctest::Approx(cx));
                    CHECK(rQOi(1) == doctest::Approx(cy));
                }
            }
        }

        GIVEN("The negative optical axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 0.0, -1.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

            WHEN("Checking field of view")
            {
                bool isWithinFOV = cam.isVectorWithinFOV(uPCc);

                THEN("Vector is outside field of view")
                {
                    CHECK_FALSE(isWithinFOV);
                }
            }
        }

        GIVEN("The positive horizontal image axis unit vector")
        {
            cv::Vec3d uPCc(1.0, 0.0, 0.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

            WHEN("Checking field of view")
            {
                bool isWithinFOV = cam.isVectorWithinFOV(uPCc);

                THEN("Vector is outside field of view")
                {
                    CHECK_FALSE(isWithinFOV);
                }
            }
        }

        GIVEN("The positive vertical image axis unit vector")
        {
            cv::Vec3d uPCc(0.0, 1.0, 0.0);
            REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

            WHEN("Checking field of view")
            {
                bool isWithinFOV = cam.isVectorWithinFOV(uPCc);

                THEN("Vector is outside field of view")
                {
                    CHECK_FALSE(isWithinFOV);
                }
            }
        }

        GIVEN("A pixel location")
        {
            cv::Vec2d rQOi_given(5.3, 8.7);
            WHEN("Evaluating pixelToVector")
            {
                cv::Vec3d uPCc = cam.pixelToVector(rQOi_given);
                REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

                THEN("Vector corresponds to given pixel")
                {
                    cv::Vec2d rQOi_actual = cam.vectorToPixel(uPCc);
                    CHECK(rQOi_actual(0) == doctest::Approx(rQOi_given(0)));
                    CHECK(rQOi_actual(1) == doctest::Approx(rQOi_given(1)));
                }
            }
        }

        GIVEN("An arbitrary world point and body pose")
        {
            cv::Vec3d rPNn(0.141886338627215, 0.421761282626275, 0.915735525189067);

            Pose<double> Tnb;
            Tnb.translationVector << 0.792207329559554, 0.959492426392903, 0.655740699156587;
            Tnb.rotationMatrix <<
                 0.988707451899469, -0.0261040852852265, 0.147567446579119,
               -0.0407096903396656,   0.900896143585899, 0.432121348216568,
                -0.144223076069359,  -0.433249022161017,  0.88966004132231;
            REQUIRE((Tnb.rotationMatrix.transpose()*Tnb.rotationMatrix - Eigen::Matrix3d::Identity()).lpNorm<Eigen::Infinity>() < 100*std::numeric_limits<double>::epsilon()); // Must be orthogonal

            WHEN("Evaluating worldToVector")
            {
                cv::Vec3d uPCc = cam.worldToVector(rPNn, Tnb);

                THEN("A unit vector is returned")
                {
                    REQUIRE(cv::norm(uPCc) == doctest::Approx(1.0));

                    AND_THEN("The expected oracle result is returned")
                    {
                        cv::Vec3d uPCc_oracle(-0.745857120799272, -0.656980343862456, -0.109881677869373);
                        CHECK(uPCc(0) == doctest::Approx(uPCc_oracle(0)));
                        CHECK(uPCc(1) == doctest::Approx(uPCc_oracle(1)));
                        CHECK(uPCc(2) == doctest::Approx(uPCc_oracle(2)));
                    }
                }
            }

            WHEN("Evaluating worldToPixel")
            {
                cv::Vec2d rQOi = cam.worldToPixel(rPNn, Tnb);

                THEN("worldToPixel matches the composition of worldToVector and vectorToPixel")
                {
                    cv::Vec2d rQOi_expected = cam.vectorToPixel(cam.worldToVector(rPNn, Tnb));
                    REQUIRE(rQOi(0) == doctest::Approx(rQOi_expected(0)));
                    REQUIRE(rQOi(1) == doctest::Approx(rQOi_expected(1)));

                    AND_THEN("The expected oracle result is returned")
                    {
                        cv::Vec2d rQOi_oracle(34.039103190885967, 36.073879427476669);
                        CHECK(rQOi(0) == doctest::Approx(rQOi_oracle(0)));
                        CHECK(rQOi(1) == doctest::Approx(rQOi_oracle(1)));
                    }
                }
            }
        }
    }
}
