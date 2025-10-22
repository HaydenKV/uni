#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include "Pose.hpp"
#include "Camera.h"
#include "SystemSLAMPointLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "Plot.h"
#include "confidence_region_demo.h"

void confidenceRegionDemo(const Camera & camera, ChessboardData & chessboardData, const std::filesystem::path & outputDirectory, int interactive)
{
    bool doExport = !outputDirectory.empty();

    // Display loaded calibration data
    camera.printCalibration();

    // Reconstruct extrinsic parameters (camera pose) for each chessboard image
    chessboardData.recoverPoses(camera);

    // Initial state mean
    Eigen::VectorXd mu(24);
    mu.setZero();

    // Landmark position mean
    const Chessboard & cb = chessboardData.chessboard;
    mu.segment(12, 3) << 0, 0, 0;
    mu.segment(15, 3) << (cb.boardSize.width - 1)*cb.squareSize, (cb.boardSize.height - 1)*cb.squareSize, 0;
    mu.segment(18, 3) << (cb.boardSize.width - 1)*cb.squareSize, 0, 0;
    mu.segment(21, 3) << 0, (cb.boardSize.height - 1)*cb.squareSize, 0;

    // Initial state square-root covariance
    Eigen::MatrixXd S(24, 24);
    S.setZero();
    S.diagonal().setConstant(1e-6);

    // Joint square-root covariance for camera pose and landmark positions
    S.bottomRightCorner(18, 18) <<
                 0.004,                 0,                 0,                 0,                 0,                 0,  -0.0001218861989,  -0.0001684482823,   0.0005186017292,  -1.121583175e-05,   4.792580983e-05,   0.0003999196878,  -7.082144231e-05,   0.0004359874182,  -0.0001235241315,   0.0001046078084,   -0.000120242372,   0.0005714317945,
                     0,             0.004,                 0,                 0,                 0,                 0,   0.0003735601624,  -0.0009440237331,   0.0006041180125,   0.0002011991299,   5.689819086e-06,   0.0004294712102,  -0.0001241695169,   0.0006195504808,   9.787516902e-05,   -1.88885821e-05,  -3.253893738e-05,  -0.0006595832305,
                     0,                 0,             0.004,                 0,                 0,                 0,   0.0002163895272,   8.325566297e-05,   0.0002068575389,   0.0003008590306,  -4.819004652e-06,  -0.0001205594835,   0.0003278497175,   0.0003074504147,   0.0002052737061,   0.0001229131583,   0.0002675065052,   0.0003532850733,
                     0,                 0,                 0,             0.004,                 0,                 0,   0.0004207496657,  -0.0004151298868,   0.0008971410386,   -0.000386735822,  -0.0002040520233,   0.0002908393883,   9.933311632e-06,   0.0001695307187,  -0.0004276746256,  -0.0001702484192,  -0.0002962109914,   -1.94101033e-05,
                     0,                 0,                 0,                 0,             0.004,                 0,   8.156385209e-05,  -7.313289049e-05,  -0.0002975041371,  -8.101766598e-05,  -0.0001538488012,  -0.0002193362193,   0.0005863788663,  -0.0001017886825,   4.035902357e-05,    0.000101309739,   0.0001765821664,   3.299114155e-05,
                     0,                 0,                 0,                 0,                 0,             0.004,   0.0002620533391,  -0.0005584474924,   0.0008183357913,   9.462487892e-05,   0.0007578768087,   1.867350967e-05,   0.0003536443669,   0.0005055334602,  -0.0001152390647,  -6.120292163e-05,   7.467459669e-05,  -0.0002890384719,
                     0,                 0,                 0,                 0,                 0,                 0,    0.004954434098,   0.0001293944502,  -0.0001564012467,  -0.0009322762908,  -0.0003707343606,   0.0005882247694,    0.001832695194,   0.0005127838734,  -0.0001536481746,   7.857706306e-05,   -0.000966662368,   0.0004314040586,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,    0.004854608645,    0.000302464133,   0.0004779242515,   -0.001719749129,   0.0004498398195,   -0.000209773774,   -0.002307812758,  -0.0004072421481,  -0.0003446071893,    0.002759572943,   0.0004224962389,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,    0.004758604155,   0.0003054931662,   0.0008301103238,    0.001654434643,    0.002230672967,    0.002108356934,   -0.001297944999,    0.001135741591,   -0.002389906904,  -0.0001359460753,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,     0.01495034842,   4.042702635e-06,  -8.343036335e-06,   0.0001594669736,  -1.652611962e-05,   0.0001504435499,   4.459738497e-05,   -0.000254619767,  -2.392750835e-05,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,    0.004535844249,  -8.650744805e-05,  -0.0002763189486,  -0.0001031186566,    0.002149450554,    0.001153417836,    0.001865309601,   -0.000974955525,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,    0.004605970922,   -0.001187228442,   0.0003047862162,   0.0006511643429,   0.0009409837974,    0.002119177446,    0.001109994536,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,    0.003809607038,   -0.001549956364,    0.001152853435,   0.0005392412011,   0.0007346609405,    0.001241398491,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,     0.01454292264,   0.0002580634287,   2.762724422e-05,   0.0008147229518,   0.0001221739628,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,    0.004049523626,    0.001073054135,  -0.0005324900014,    0.000195422118,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,    0.004455890451,  -7.292307362e-05,  -0.0001515464566,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,   0.0009998014985,   -0.002569854609,
                     0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,                 0,     0.01460277367;


    // Initialise system
    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
    SystemSLAMPointLandmarks system(p0);

    // Initialise plot
    Plot plot(camera);

    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        // Get the body pose
        Pose<double> Tnb = camera.cameraToBody(chessboardImage.Tnc);
        const Eigen::Vector3d & rBNn = Tnb.translationVector;
        const Eigen::Matrix3d & Rnb = Tnb.rotationMatrix;
        Eigen::Vector3d Thetanb = rot2rpy(Rnb);

        // Update body pose mean
        mu.segment(6, 3) << rBNn;
        mu.segment(9, 3) << Thetanb;
        system.density = GaussianInfo<double>::fromSqrtMoment(mu, S);

        // Get a copy of the image for plot to draw on
        system.view() = chessboardImage.image.clone();

        Eigen::Matrix<double, 2, Eigen::Dynamic> Y; // dummy feature bundle
        MeasurementPointBundle measurement(0.0, Y, camera);

        // Set local copy of system and measurement for plot to use
        plot.setData(system, measurement);

        // Update plot
        plot.render();

        std::filesystem::path outputPath;
        if (doExport)
        {
            std::string outputFilename = chessboardImage.filename.stem().string()
                                       + "_out"
                                       + chessboardImage.filename.extension().string();
            outputPath = outputDirectory / outputFilename;
            cv::imwrite(outputPath.string(), plot.getFrame());
        }
        bool isLastImage = (&chessboardImage == &chessboardData.chessboardImages.back());
        if (interactive == 2 || (interactive == 1 && isLastImage))
        {
            // Start handling plot GUI events (blocking)
            plot.start();
        }
    }
}
