#include <doctest/doctest.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/eigen.hpp>
#include "../../src/Camera.h"

SCENARIO("Lens distortion")
{
    GIVEN("A camera with lens distortion")
    {
        const std::filesystem::path cameraPath = "test/data/camera2.xml";
        REQUIRE(std::filesystem::exists(cameraPath));

        // Load camera calibration
        Camera camera;
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        REQUIRE(fs.isOpened());
        fs["camera"] >> camera;

        REQUIRE(camera.cameraMatrix.rows == 3);
        REQUIRE(camera.cameraMatrix.cols == 3);
        REQUIRE(camera.cameraMatrix.type() == CV_64F);
        REQUIRE(camera.distCoeffs.cols == 1);
        REQUIRE(camera.distCoeffs.type() == CV_64F);

        GIVEN("A test pixel rQOi")
        {
            Eigen::Vector2d rQOi(2, 1);
            rQOi << 0.25*camera.imageSize.width, 0.25*camera.imageSize.height;
            
            WHEN("Calling undistortPoints")
            {
                std::vector<cv::Point2d> rQOi_cv = {cv::Point2d(rQOi(0), rQOi(1))};
                std::vector<cv::Point2d> rQbarOi_cv;
                // TODO: Lab 10
                // Task 4b: Use cv::undistortPoints to perform mapping r^i_{Q/O} -> r^i_{Q/O}_bar
                // where r^i_{Q/O} is distorted pixel location and r^i_{Q/O}_bar is undistorted pixel location
                // 
                // By passing cameraMatrix as the 6th parameter (P), we get undistorted PIXEL coordinates
                // instead of normalized coordinates.
                // Call cv::undistortPoints with P=K to get undistorted PIXEL coordinates
                cv::undistortPoints(rQOi_cv, rQbarOi_cv, camera.cameraMatrix, camera.distCoeffs, cv::noArray(), camera.cameraMatrix);

                REQUIRE(rQbarOi_cv.size() == 1);
                THEN("rQbarOi has the correct pixel coordinates")
                {
                    CHECK(std::abs(rQbarOi_cv[0].x - 393.0) < 0.1);
                    CHECK(std::abs(rQbarOi_cv[0].y - 294.8) < 0.1);
                }

                double fx = camera.cameraMatrix.at<double>( 0,  0);
                double fy = camera.cameraMatrix.at<double>( 1,  1);
                double cx = camera.cameraMatrix.at<double>( 0,  2);
                double cy = camera.cameraMatrix.at<double>( 1,  2);

                // Solve K*rPCc = pQbarOi for rPCc
                Eigen::Vector3d rPCc;

                // Convert to normalized coordinates using K^-1
                // Task 4c: Solve K*r^c_{P/C} = p for r^c_{P/C}
                // where p is homogeneous representation of r^i_{Q/O}_bar (undistorted pixel coords)
                // 
                // Step 1: Apply K^-1 to convert from undistorted pixels to normalized image coordinates
                double x_norm = (rQbarOi_cv[0].x - cx) / fx;  // Subtract principal point, divide by focal length
                double y_norm = (rQbarOi_cv[0].y - cy) / fy;

                // Step 2: Form 3D ray in camera coordinates with arbitrary magnitude (z=1)
                // This represents the bearing vector to the feature point
                rPCc << x_norm, y_norm, 1.0;

                AND_WHEN("rQOi = camera.vectorToPixel(rPCc)")
                {
                    Eigen::Vector2d rQOi_actual = camera.vectorToPixel(rPCc);

                    THEN("rQOi matches the test point")
                    {
                        CHECK(std::abs(rQOi_actual(0) - rQOi(0)) < 0.07);
                        CHECK(std::abs(rQOi_actual(1) - rQOi(1)) < 0.07);
                    }
                }
            }
        }
    }
}
