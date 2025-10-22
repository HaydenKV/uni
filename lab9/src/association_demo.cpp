#include <cassert>
#include <cstddef>
#include <algorithm>
#include <print>
#include <vector>
#include <numeric>

#include <Eigen/Core>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Camera.h"
#include "GaussianInfo.hpp"
#include "image_features.h"
#include "SystemSLAM.h"
#include "SystemSLAMPointLandmarks.h"
#include "plot_util.h"
#include "rotation.hpp"
#include "Pose.hpp"
#include "MeasurementSLAMPointBundle.h"
#include "association_util.h"
#include "association_demo.h"

// Forward declarations
static SystemSLAMPointLandmarks exampleSystemFromChessboardImage(const Camera & camera, const ChessboardImage & chessboardImage);

// ------------------------------------------------------------------------------------------
// 
// associationDemo
// 
// ------------------------------------------------------------------------------------------
cv::Mat associationDemo(const Camera & camera, const ChessboardImage & chessboardImage)
{
    SystemSLAMPointLandmarks system = exampleSystemFromChessboardImage(camera, chessboardImage);

    // ------------------------------------------------------------------------
    // Feature detector
    // ------------------------------------------------------------------------
    int maxNumFeatures = 10000;
    std::vector<PointFeature> features = detectFeatures(chessboardImage.image, maxNumFeatures);
    std::println("{} features found in {}", features.size(), chessboardImage.filename.string());
    assert(features.size() > 0);
    assert(features.size() <= maxNumFeatures);
    
    // ------------------------------------------------------------------------
    // Populate measurement set Y with elements of the features vector
    // ------------------------------------------------------------------------
    Eigen::Matrix<double, 2, Eigen::Dynamic> Y(2, features.size());
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        Y.col(i) << features[i].x, features[i].y;
    }
    double time = 0.0;
    MeasurementPointBundle measurement(time, Y, camera);
    
    // ------------------------------------------------------------------------
    // Select landmarks expected to be within the field of view of the camera
    // ------------------------------------------------------------------------
    std::vector<std::size_t> idxLandmarks;
    idxLandmarks.reserve(system.numberLandmarks());  // Reserve maximum possible size to avoid reallocation
    for (std::size_t j = 0; j < system.numberLandmarks(); ++j)
    {
        Eigen::Vector3d murPNn = system.landmarkPositionDensity(j).mean();
        cv::Vec3d rPNn;
        cv::eigen2cv(murPNn, rPNn);
        Pose Tnb = camera.cameraToBody(chessboardImage.Tnc);
        if (camera.isWorldWithinFOV(rPNn, Tnb))
        {
            std::println("Landmark {} is expected to be within camera FOV", j);
            idxLandmarks.push_back(j);
        }
        else
        {
            std::println("Landmark {} is NOT expected to be within camera FOV", j);
        }
    }

    // ------------------------------------------------------------------------
    // Run data association
    // ------------------------------------------------------------------------
    const std::vector<int> & idxFeatures = measurement.associate(system, idxLandmarks);

    // ------------------------------------------------------------------------
    // Visualisation and console output
    // ------------------------------------------------------------------------
    plotAllFeatures(system.view(), Y);
    for (std::size_t jj = 0; jj < idxFeatures.size(); ++jj)
    {
        int j           = idxLandmarks[jj];     // Index of landmark in state vector
        int i           = idxFeatures[jj];      // Index of feature
        bool isMatch    = i >= 0;

        GaussianInfo<double> featureDensity = measurement.predictFeatureDensity(system, j);
        const Eigen::VectorXd & murQOi = featureDensity.mean();

        // Plot confidence ellipse and landmark index text
        Eigen::Vector3d colour = isMatch ? Eigen::Vector3d(0, 0, 255) : Eigen::Vector3d(255, 0, 0);
        plotGaussianConfidenceEllipse(system.view(), featureDensity, colour);
        plotLandmarkIndex(system.view(), murQOi, colour, j);

        if (isMatch)
            std::println("Feature {} located at [{} {}] in image matches landmark {}.", i, Y.col(i)(0), Y.col(i)(1), j);
        else
            std::println("No feature associated with landmark {}.", j);
    }
    plotMatchedFeatures(system.view(), idxFeatures, Y);
    std::println("");

    return system.view();
}

// ----------------------------------------------------------------------------
// Helper functions (shouldn't need to edit)
// ----------------------------------------------------------------------------
SystemSLAMPointLandmarks exampleSystemFromChessboardImage(const Camera & camera, const ChessboardImage & chessboardImage)
{
    // Body velocity mean
    Eigen::Vector6d munu;
    munu.setZero();

    // Obtain body pose from camera pose
    Pose<double> Tnb = camera.cameraToBody(chessboardImage.Tnc);
    const Eigen::Vector3d & rBNn = Tnb.translationVector;
    const Eigen::Matrix3d & Rnb = Tnb.rotationMatrix;
    Eigen::Vector3d Thetanb = rot2rpy(Rnb);

    // Body pose mean
    Eigen::Vector6d mueta;
    mueta << rBNn, Thetanb;

    // Landmark mean
    Eigen::VectorXd murPNn(27);
    murPNn <<
                        -0.2,
                         0.1,
                           0,
                           0,
                        -0.3,
                           0,
                           0,
                         0.6,
                           0,
                         0.4,
                         0.6,
                           0,
                         0.4,
                        -0.4,
                           0,
                       0.088,
                       0.066,
                           0,
                       0.022,
                       0.132,
                           0,
                       0.297,
                           0,
                           0,
                     0.29029,
                     0.09502,
                    -0.14843;

    // State mean
    Eigen::VectorXd mu(6 + 6 + 27);
    mu << munu, mueta, murPNn;

    // State square-root covariance
    Eigen::MatrixXd S(39, 39);
    S.setZero();
    S.diagonal().setConstant(1e-6);

    S.bottomRightCorner(33, 33) <<
         0.001427830283,    0.001229097682,    0.003320690394,   -0.002585376784,    0.004103269664,   -0.002795866555,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,
                      0,   0.0003924364969,    0.001926923125,    0.004464152406,    -0.00231243553,   0.0001452205508,                 0,                 0,                -0,                -0,                -0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                 0,                -0,                -0,
                      0,                 0,   0.0005341369799,   0.0006053459295,   0.0009182405056,    -0.00099283847,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                -0,                 0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                 0,
                      0,                 0,                 0,    0.001328048581,     0.00308475703,  -0.0007088613796,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,     0.00300904338,   -0.001734500276,                 0,                 0,                 0,                -0,                -0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,
                      0,                 0,                 0,                 0,                 0,    0.002043219303,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0045,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,             0.006,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0045,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,                -0,                 0,                -0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0045,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,             0.006,                 0,                -0,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,                -0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0045,                 0,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015,                 0,
                      0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,            0.0015;

    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
    SystemSLAMPointLandmarks system(p0);
    system.view() = chessboardImage.image.clone();
    return system;
}
