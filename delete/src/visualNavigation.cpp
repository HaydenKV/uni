#include <filesystem>
#include <string>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <array>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include <ctime>
#include <csignal>
#include <atomic>
#include <print>
#include <cstdlib>
#include <numbers> 
#include <algorithm>

#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "Camera.h"
#include "Pose.hpp"
#include "Plot.h"
#include "SystemSLAMPointLandmarks.h"
#include "SystemSLAMPoseLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "MeasurementSLAMDuckBundle.h"
#include "DuckDetectorONNX.h"
#include "GaussianInfo.hpp"
#include "imagefeatures.h"
#include "rotation.hpp"

// ============================================================================
// SCENARIO 1 CONSTANTS (Shared across the file)
// ============================================================================
namespace {
    constexpr float  TAG_SIZE_METERS = 0.166f;       // ArUco tag edge length (166mm)
    constexpr double REPROJ_ERR_THRESH_PX = 3.0;     // IPPE reprojection gate (pixels)
    
    // Typical PnP accuracy at 2–3 m: ≈5–10 cm position, ≈3–5° orientation.
    constexpr double INIT_POS_SIGMA = 0.1;                             // [m]
    constexpr double INIT_ANG_SIGMA = (5.0 * std::numbers::pi / 180.0); // [rad]
    
    // Small jitter avoids perfect init → supports stable Hessian building.
    constexpr double INIT_POS_OFFSET = 0.02;  // [m]
}

// ============================================================================
// Async-signal-safe shutdown flag + handler (handler only flips an atomic)
// ============================================================================
static std::atomic<bool> g_stop{false};
static void onSignal(int) { g_stop.store(true, std::memory_order_relaxed); }

// Helper: convert Rodrigues rvec to rotation matrix
static inline Eigen::Matrix3d rodriguesToRot(const cv::Vec3d& rvec)
{
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r,c) = Rcv.at<double>(r,c);
    return R;
}

/*
runVisualNavigationFromVideo:
Main loop orchestrating model (4)–(5) and measurement updates.

Model (shared across scenarios):
  x = [ν(6), r^n_{B/N}(3), Θ^n_B(3), m...] with   dη/dt = J_K(η)ν,   dm/dt = 0  (4)–(5).

Scenario 1 measurement (unique tags):
  Landmark m_j = [ r^n_{j/N}, Θ^n_j ]^T (6);  3D corners from tag frame via (8)–(9);
  image mapping uses u = π(K,dist, r^c) with association by tag ID; log-likelihood (7).
*/
void runVisualNavigationFromVideo(
    const std::filesystem::path& videoPath,
    const std::filesystem::path& cameraPath,
    int scenario,
    int interactive,
    const std::filesystem::path& outputDirectory,
    int max_frames)
{
    assert(!videoPath.empty());

    // Register Ctrl-C / SIGTERM (sets a flag checked in the loop)
    g_stop.store(false, std::memory_order_relaxed);
    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    // ---------- Output setup ----------
    std::filesystem::path outputPath;
    const bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // ---------- Load camera (K, dist, T_bc) ----------
    Camera camera;
    {
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "File: " << cameraPath << " does not exist or cannot be opened\n";
            std::exit(EXIT_FAILURE);
        }
        fs["camera"] >> camera;
        fs.release();
    }
    camera.printCalibration();
    
    // ---------- Open video ----------
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Total frames in video: " << nFrames << std::endl;
    std::cout << "Video duration (approx): " << (nFrames / fps) << " seconds" << std::endl;

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    // Ensure camera.imageSize is set (used by Plot/image gating)
    {
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (camera.imageSize.width <= 0 || camera.imageSize.height <= 0)
            camera.imageSize = cv::Size(w, h);
    }

    // ---------- Plot & export ----------
    Plot plot(camera);
    const cv::Size plotSize = plot.renderSize();

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);

    if (doExport) {
        const int codec = cv::VideoWriter::fourcc('m','p','4','v');
        if (!videoOut.open(outputPath.string(), codec, fps, plotSize)) {
            std::cerr << "Failed to open video for writing: " << outputPath << "\n";
        } else {
            bufferedVideoWriter.start(videoOut);
        }
    }

    // ---------- Build system state ----------
    std::unique_ptr<DuckDetectorONNX> duckDetector;
    std::unique_ptr<SystemSLAM> systemPtr;
    if (scenario == 1)
    {
        // Body-only prior (first 12 states): ν(6), r(3), Θ(3)
        Eigen::VectorXd mu_body(12);
        mu_body.setZero();
        mu_body.segment<3>(6) << 0.0, 0.0, -1.6; // r^n_{B/N}
        mu_body.segment<3>(9) << 0.0, 0.0, 0.0; // Θ^n_B

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12);
        S_body.block<3,3>(0,0) *= 1.0;          // v uncertainty
        S_body.block<3,3>(3,3) *= 0.5;         // ω uncertainty
        const double d2r = (1.0 * std::numbers::pi / 180.0);
        S_body.block<3,3>(6,6) *= 0.5;         // r uncertainty (≈1 cm)
        S_body.block<3,3>(9,9) *= (1 * d2r);    // Θ uncertainty (≈1°)

        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPoseLandmarks>(SystemSLAMPoseLandmarks(p0));
    }
    else if (scenario == 2) {
        // ---- Point-landmark SLAM system (we won't update it yet) ----
        Eigen::VectorXd mu_body(12);  mu_body.setZero();        // ν(6), r(3), Θ(3)
        mu_body.segment<3>(6) << 0.0, 0.0, -1.0; // r^n_{B/N}
        mu_body.segment<3>(9) << 0.0, 0.0, 0.0; // Θ^n_B

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12);
        S_body.block<3,3>(0,0) *= 0.5;          // v uncertainty
        S_body.block<3,3>(6,6) *= 0.5;                         // position sqrt-cov
        const double d2r = (1.0 * std::numbers::pi / 180.0);
        S_body.block<3,3>(9,9) *= (5.0 * d2r);                    // orientation sqrt-cov
        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(SystemSLAMPointLandmarks(p0));

        // ---- ONNX Duck detector (same path as Lab 3) ----
        const std::filesystem::path onnx_file = "../src/duck_with_postprocessing.onnx";
        assert(std::filesystem::exists(onnx_file) && "[DuckDetector] ONNX model not found at ../src/duck_with_postprocessing.onnx");
        duckDetector = std::make_unique<DuckDetectorONNX>(onnx_file.string());
        assert(duckDetector && "[DuckDetector] construction failed");
    }
    else
    {
        assert("broken if you're here");
        // Placeholder for other scenarios (unchanged).
        Eigen::VectorXd mu(24);
        mu.setZero();
        mu.segment<3>(12) << 0.0, 0.0, 0.0;
        mu.segment<3>(15) << 1.0, 0.0, 0.0;
        mu.segment<3>(18) << 1.0, 1.0, 0.0;
        mu.segment<3>(21) << 0.0, 1.0, 0.0;
        Eigen::MatrixXd S = Eigen::MatrixXd::Identity(24,24) * 1e-3;
        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(SystemSLAMPointLandmarks(p0));
    }
    SystemSLAM& system = *systemPtr;

    // Prime plot with an empty measurement
    {
        Eigen::Matrix<double,2,Eigen::Dynamic> Y0(2,0);
        MeasurementPointBundle m0(0.0, Y0, camera);
        plot.setData(system, m0);
    }

    // Persistent ID/association (Scenario 1)
    static std::vector<int> id_by_landmark;                 // landmark idx → tag ID
    static std::unordered_map<int, std::size_t> id2lm;      // tag ID → landmark idx
    id_by_landmark.clear();
    id2lm.clear();
    
    // ============================== MAIN LOOP ==============================
    int frameIdx = 0;
    while (true)
    {
        // Graceful shutdown (Ctrl-C / SIGTERM)
        if (g_stop.load(std::memory_order_relaxed)) {
            std::cout << "\n[INFO] Caught shutdown signal — finishing and closing video...\n";
            break;
        }

        // Optional frame cap
        if (max_frames > 0 && frameIdx >= max_frames) {
            std::cout << "[INFO] Reached max_frames=" << max_frames << " — stopping.\n";
            break;
        }

        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty()) {
            std::cout << "[INFO] End of video stream — stopping.\n";
            break;
        }

        const double t = (fps > 0.0) ? (frameIdx / fps) : frameIdx;

        if (scenario == 1)
        {
            // ----------------------------------------------------------------
            // STEP 1: Detect ArUco tags + IPPE pose + reprojection gating.
            // Measurement will later assemble Y = [u1..u4,v1..v4] per tag.
            // ----------------------------------------------------------------
            std::vector<cv::Vec3d> rvecs, tvecs;
            std::vector<double> meanErrs;
            ArucoDetections dets = detectArUcoPOSE(
                imgin,
                cv::aruco::DICT_6X6_250,
                /*doCornerRefine*/ true,
                camera.cameraMatrix,
                camera.distCoeffs,
                TAG_SIZE_METERS,
                &rvecs, &tvecs, &meanErrs,
                REPROJ_ERR_THRESH_PX,
                /*drawRejected*/ false
            );

            auto* sysPose = dynamic_cast<SystemSLAMPoseLandmarks*>(&system);
            assert(sysPose && "Scenario 1 expects SystemSLAMPoseLandmarks");

            if (id_by_landmark.size() < system.numberLandmarks())
                id_by_landmark.resize(system.numberLandmarks(), -1);

            // Camera mean pose (used for world placement of new landmarks)
            const Eigen::VectorXd xmean = sysPose->density.mean();
            const Eigen::Vector3d rCNn  = SystemSLAM::cameraPosition(camera, xmean);
            const Eigen::Matrix3d Rnc   = SystemSLAM::cameraOrientation(camera, xmean);

            // ----------------------------------------------------------------
            // STEP 2: Initialize new landmarks (IDs unseen so far).
            // Uses (8)–(9) to place corners via tag pose; stores landmark pose (6).
            // ----------------------------------------------------------------
            int nInitialized = 0;
            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                const int tagId = dets.ids[i];

                // GATE 1: only new IDs
                if (id2lm.find(tagId) != id2lm.end()) continue;

                // GATE 2: all 4 corners inside with a safety margin (robust association)
                if (!camera.areCornersInside(dets.corners[i], CamDefaults::BorderMarginPx)) continue;

                // GATE 3: PnP in front of camera
                if (tvecs[i][2] <= 1e-3) continue;

                // IPPE pose (camera→tag) → world
                const Eigen::Matrix3d Rcj = rodriguesToRot(rvecs[i]);
                const Eigen::Vector3d rCjVec(tvecs[i][0], tvecs[i][1], tvecs[i][2]);
                const Eigen::Matrix3d Rnj = Rnc * Rcj;
                Eigen::Vector3d rnj = rCNn + Rnc * rCjVec;
                const Eigen::Vector3d Thetanj = rot2rpy(Rnj);

                // Small random jitter in position to avoid perfect init
                std::srand(static_cast<unsigned>(std::time(nullptr)) + static_cast<unsigned>(i));
                auto jitter = [](double off){ return off * (2.0 * (std::rand() / (double)RAND_MAX) - 1.0); };
                rnj(0) += jitter(INIT_POS_OFFSET);
                rnj(1) += jitter(INIT_POS_OFFSET);
                rnj(2) += jitter(INIT_POS_OFFSET);

                // Initial sqrt-covariance (diagonal) for pose landmark
                Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Zero();
                Sj(0,0) = INIT_POS_SIGMA;
                Sj(1,1) = INIT_POS_SIGMA;
                Sj(2,2) = INIT_POS_SIGMA;
                Sj(3,3) = INIT_ANG_SIGMA;
                Sj(4,4) = INIT_ANG_SIGMA;
                Sj(5,5) = INIT_ANG_SIGMA;

                const std::size_t j = sysPose->appendLandmark(rnj, Thetanj, Sj);

                if (j >= id_by_landmark.size()) id_by_landmark.resize(j+1, -1);
                id_by_landmark[j] = tagId;
                id2lm[tagId] = j;
                
                nInitialized++;
            }

            // ----------------------------------------------------------------
            // STEP 3: Build measurement matrix Y (2 × 4N) from detected corners.
            // Likelihood uses per-corner Gaussian with missed-detection penalty (7).
            // ----------------------------------------------------------------
            const std::size_t N = dets.ids.size();
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, 4*N);
            for (std::size_t i = 0; i < N; ++i) {
                const auto& c = dets.corners[i];            // TL, TR, BR, BL
                for (int k = 0; k < 4; ++k) {
                    Y(0, 4*i + k) = c[k].x;
                    Y(1, 4*i + k) = c[k].y;
                }
            }
            assert(Y.cols() == 4 * static_cast<int>(N) && "Y must pack 4 columns per tag");

            // ----------------------------------------------------------------
            // STEP 4: Time update + measurement update (EKF/InfoFilter policy).
            // ----------------------------------------------------------------
            system.view() = dets.annotated.empty() ? imgin : dets.annotated;

            MeasurementSLAMUniqueTagBundle meas(t, Y, camera, dets.ids);
            meas.setIdByLandmark(id_by_landmark);
            meas.process(system);                 // propagate (4)–(5) + correct (7)
            id_by_landmark = meas.idByLandmark(); // persist any mapping updates

            // ----------------------------------------------------------------
            // STEP 5: Diagnostics every 10 frames (counts, camera state, σ)
            // ----------------------------------------------------------------
            if (frameIdx % 10 == 0) {
                std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
                std::cout << "║ Frame " << std::setw(5) << frameIdx 
                          << " | Time: " << std::fixed << std::setprecision(2) << t << "s"
                          << std::string(30, ' ') << "║\n";
                std::cout << "╠════════════════════════════════════════════════════════════╣\n";
                
                // Detection stats
                std::cout << "║ DETECTIONS                                                 ║\n";
                std::cout << "║   Total landmarks in map:   " << std::setw(4) << system.numberLandmarks() << "                                  ║\n";
                std::cout << "║   Tags detected this frame: " << std::setw(4) << dets.ids.size()          << "                                  ║\n";
                std::cout << "║   New landmarks initialized:" << std::setw(4) << nInitialized             << "                                  ║\n";
                
                // Association stats
                int nAssoc = 0;
                for (int idx : meas.idxFeatures()) if (idx >= 0) ++nAssoc;
                std::cout << "║   Landmarks associated:     " << std::setw(4) << nAssoc                   << "                                  ║\n";
                std::cout << "╠════════════════════════════════════════════════════════════╣\n";
                
                // Camera state
                const Eigen::VectorXd x = system.density.mean();
                const Eigen::Vector3d camPos   = SystemSLAM::cameraPosition(camera, x);
                const Eigen::Vector3d camVel   = x.segment<3>(0);
                const Eigen::Vector3d camOmega = x.segment<3>(3);
                
                std::cout << "║ CAMERA STATE                                               ║\n";
                std::cout << "║   Position (m):    [" 
                          << std::setw(7) << std::fixed << std::setprecision(3) << camPos(0) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camPos(1) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camPos(2) << "]   ║\n";
                std::cout << "║   Velocity (m/s):  [" 
                          << std::setw(7) << std::fixed << std::setprecision(3) << camVel(0) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camVel(1) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camVel(2) << "]   ║\n";
                std::cout << "║   Ang vel (rad/s): [" 
                          << std::setw(7) << std::fixed << std::setprecision(3) << camOmega(0) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camOmega(1) << ", "
                          << std::setw(7) << std::fixed << std::setprecision(3) << camOmega(2) << "]   ║\n";
                
                // Landmark σ (position marginals)
                const std::size_t nShow = system.numberLandmarks();
                if (nShow > 0) {
                    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
                    std::cout << "║ LANDMARK UNCERTAINTIES (Show " << nShow << " landmarks)                  ║\n";
                    for (std::size_t i = 0; i < nShow; ++i) {
                        const auto posDen = system.landmarkPositionDensity(i);
                        const Eigen::Vector3d pos_mean = posDen.mean();
                        const double std_x = std::abs(posDen.marginal(Eigen::seqN(0,1)).sqrtCov()(0,0));
                        const double std_y = std::abs(posDen.marginal(Eigen::seqN(1,1)).sqrtCov()(0,0));
                        const double std_z = std::abs(posDen.marginal(Eigen::seqN(2,1)).sqrtCov()(0,0));
                        
                        std::cout << "║   LM[" << i << "] pos: ["
                                  << std::setw(6) << std::fixed << std::setprecision(2) << pos_mean(0) << ","
                                  << std::setw(6) << std::fixed << std::setprecision(2) << pos_mean(1) << ","
                                  << std::setw(6) << std::fixed << std::setprecision(2) << pos_mean(2) << "]m  ";
                        std::cout << "σ:["
                                  << std::setw(5) << std::fixed << std::setprecision(3) << std_x << ","
                                  << std::setw(5) << std::fixed << std::setprecision(3) << std_y << ","
                                  << std::setw(5) << std::fixed << std::setprecision(3) << std_z << "]m ║\n";
                    }
                }
                std::cout << "╚════════════════════════════════════════════════════════════╝\n";
            }

            // Visualization — left/right panes per assignment spec
            plot.setData(system, meas);
        }
        else if (scenario == 2)
        {
            auto* sysPts = dynamic_cast<SystemSLAMPointLandmarks*>(&system);
            assert(sysPts && "Scenario 2 expects SystemSLAMPointLandmarks");
            assert(duckDetector && "duckDetector must be initialised in scenario 2");

            // DEBUG: Print frame number
            printf("\n--- [FRAME %d at t=%.3fs] ---\n", frameIdx, t);

            cv::Mat annotated = duckDetector->detect(imgin);
            system.view() = annotated.empty() ? imgin : annotated;

            const auto& C = duckDetector->last_centroids();
            const auto& A = duckDetector->last_areas();
            assert(C.size() == A.size());

            const int N = static_cast<int>(C.size());
            printf("[Nav] DuckDetector found %d ducks.\n", N);


            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, N);
            Eigen::VectorXd Avec(N);
            for (int i = 0; i < N; ++i) {
                Y(0,i) = static_cast<double>(C[i].x);
                Y(1,i) = static_cast<double>(C[i].y);
                Avec(i) = static_cast<double>(A[i]);
            }

            constexpr double SIGMA_C_PX  = 15.0;
            constexpr double SIGMA_A_PX2 = 150.0;
            constexpr double DUCK_R_M    = 0.07;

            MeasurementSLAMDuckBundle meas(t, Y, Avec, camera, SIGMA_C_PX, SIGMA_A_PX2, DUCK_R_M);

            if (frameIdx == 0) {
                printf("[Nav] First frame, initializing all %d detections as landmarks.\n", N);
                size_t added = sysPts->appendFromDuckDetections(
                    camera, Y, Avec,
                    camera.cameraMatrix.at<double>(0,0), // fx
                    camera.cameraMatrix.at<double>(1,1), // fy
                    DUCK_R_M,
                    /*pos_sigma_m*/ 0.30 // Initial uncertainty
                );
                printf("[Nav] Added %zu new landmarks. Total landmarks in map: %zu\n", added, sysPts->numberLandmarks());
            } else {
                std::vector<std::size_t> idxLandmarks(sysPts->numberLandmarks());
                std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);

                if (!idxLandmarks.empty()) {
                    meas.associate(*sysPts, idxLandmarks);
                }
                const auto& idxFeats = meas.idxFeatures();

                std::vector<int> surplusIdx;
                std::vector<bool> featMatched(N, false);
                for (int f : idxFeats) {
                    if (f >= 0) {
                        featMatched[f] = true;
                    }
                }
                for (int i = 0; i < N; ++i) {
                    if (!featMatched[i]) {
                        surplusIdx.push_back(i);
                    }
                }

                printf("[Nav] Surplus detections to initialize: %zu\n", surplusIdx.size());
                if (!surplusIdx.empty()) {
                    const int S = static_cast<int>(surplusIdx.size());
                    Eigen::Matrix<double,2,Eigen::Dynamic> Ysurp(2, S);
                    Eigen::VectorXd Asurp(S);
                    for (int k = 0; k < S; ++k) {
                        const int i = surplusIdx[k];
                        Ysurp.col(k) = Y.col(i);
                        Asurp(k) = Avec(i);
                    }
                    size_t added = sysPts->appendFromDuckDetections(
                        camera, Ysurp, Asurp,
                        camera.cameraMatrix.at<double>(0,0), // fx
                        camera.cameraMatrix.at<double>(1,1), // fy
                        DUCK_R_M,
                        0.30
                    );
                    printf("[Nav] Added %zu new landmarks from surplus. Total landmarks in map: %zu\n", added, sysPts->numberLandmarks());
                }
            }

            printf("[Nav] Calling meas.process()...\n");
            meas.process(system);
            printf("[Nav] Calling plot.setData()...\n");
            plot.setData(system, meas);
        }

        else
        {
            assert("broken if you're here");
            // Other scenarios unchanged in this file
            system.view() = imgin;
            Eigen::Matrix<double,2,Eigen::Dynamic> Ynow(2,0);
            MeasurementPointBundle meas(t, Ynow, camera);
            plot.setData(system, meas);
        }

        plot.render();
        if (doExport) bufferedVideoWriter.write(plot.getFrame());

        if (interactive == 2 || (interactive == 1 && (--nFrames == 0)))
            plot.start();

        frameIdx++;
    }

    // ---------- Clean shutdown (finalize export and threads) ----------
    if (doExport) {
        bufferedVideoWriter.stop();
        videoOut.release();
    }
    bufferedVideoReader.stop();
}
