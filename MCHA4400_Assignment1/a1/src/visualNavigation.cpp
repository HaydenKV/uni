#include <filesystem>
#include <string>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <numbers>
#include <algorithm>
#include <cmath>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

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

/*
NOTE:
    Scenario 1: ATTEMPTED
    Scenario 2: Visiualisation should appear 'better' than video in output folder, its the same solution  - unfortunately I didn't time to rerun it as it takes all night
    Scenario 3: ATTEMPTED

*/


// ============================================================================
// Tunable parameters
// ============================================================================
namespace {

    // ----- Scenario 1 (tags) -----
    constexpr float  TAG_SIZE_METERS        = 0.166f;                 // ArUco tag edge (166 mm)
    constexpr double REPROJ_ERR_THRESH_PX   = 2.0;                    // IPPE reprojection gate (px)
    constexpr double INIT_POS_SIGMA_TAG     = 0.5;                    // [m] 1σ for tag position init
    constexpr double INIT_ANG_SIGMA_TAG     = (1.0 * std::numbers::pi / 180.0); // [rad]
    constexpr double INIT_POS_OFFSET        = 0.02;                   // [m] small jitter for stable Hessian

    // ----- Scenario 2 (ducks) -----
    // Measurement model and association
    constexpr double DUCK_RADIUS_M          = 0.054;                  // physical radius [m]
    constexpr int    NMAX_LANDMARKS         = 200;                    // cap on map size
    constexpr int    MAX_NEW_PER_FR         = 6;                      // births per frame
    constexpr int    NMS_MIN_SEP_PX         = 40;                     // seed spacing (px)
    constexpr double SIGMA_PX_MEAS          = 2.5;                    // pixel noise (centroid)
    constexpr double SIGMA_AREA_MEAS        = 3000.0;                 // area noise (de-emphasise area)
    constexpr int    CULL_FAIL_THRESH       = 6;                      // consecutive misses before cull

    // HSV yellow gate + area filter
    constexpr int    H_MIN_YELLOW           = 15;                     // HSV hue min
    constexpr int    H_MAX_YELLOW           = 35;                     // HSV hue max
    constexpr int    S_MIN_YELLOW           = 60;                     // HSV saturation min
    constexpr int    V_MIN_YELLOW           = 60;                     // HSV value min
    constexpr double MIN_YELLOW_RATIO       = 0.04;                   // min ratio inside circular ROI
    constexpr double MIN_AREA_PX            = 100.0;                  // reject tiny specks

    // Seeding from centroid+area
    constexpr double Z_MIN_SEED             = 0.40;                   // [m] clamp
    constexpr double Z_MAX_SEED             = 6.00;                   // [m] clamp
    constexpr double REPROJ_SEED_THRESH_PX  = 25.0;                   // back-projection gate (px)
    constexpr double EXISTING_SEED_BLOCK_PX = 30.0;                   // block near existing LM projection (px)
    constexpr double SIGMA_PERP_SEED        = 0.15;                   // [m] across-ray 1σ
    constexpr double SIGMA_DEPTH_SEED       = 0.50;                   // [m] along-ray  1σ

    // ----- Scenario 3 (points) -----
    // Detection & ranking
    constexpr FeatureDetector S3_DETECTOR        = FeatureDetector::ShiTomasi; // Or FAST
    constexpr int    S3_FEAT_MAXN                = 2000;   // Features to detect per frame
    // Birth control
    constexpr int    S3_BIRTHS_PER_FRAME         = 10;     // Max newborn landmarks per frame (strict)
    constexpr int    S3_BOOTSTRAP_BIRTHS         = 10;     // Max births when map is empty (strict)
    constexpr int    S3_NMAX_LANDMARKS           = 300;    // Global landmark cap
    constexpr double S3_MIN_SEP_PX               = 40.0;   // Birth NMS spacing (px)
    constexpr double S3_REPROJ_GATE_PX           = 20.0;   // Birth back-projection gate (px)
    // Seeding model
    constexpr double S3_DEPTH_GUESS_M            = 1.6;    // Fixed initial depth (m)
    constexpr double S3_SIGMA_PERP_M             = 0.20;   // Across-ray 1σ (m)
    constexpr double S3_SIGMA_ALONG_M            = 1.00;   // Along-ray 1σ (m)
    // Deletion policy (visibility-aware)
    constexpr int    S3_CULL_FAIL_THRESH         = 10;     // Visible-but-unmatched frames before cull
    // Merge (deduplicate) policy
    constexpr double S3_MERGE_DIST_M             = 0.50;   // Merge if two landmarks are closer than this in 3D (m)
    constexpr double S3_MERGE_PIX_PX             = 100.0;  // And their predicted pixels are within this distance (px)
    constexpr int    S3_MERGE_MAX_PER_FRAME      = 10;     // Max merges to attempt per frame



}

// ============================================================================
// Async-signal-safe shutdown flag + handler (handler only flips an atomic)
// ============================================================================
static std::atomic<bool> g_stop{false};
static void onSignal(int) { g_stop.store(true, std::memory_order_relaxed); }

// Helper: convert Rodrigues rvec to rotation matrix
[[nodiscard]] static inline Eigen::Matrix3d rodriguesToRot(const cv::Vec3d& rvec)
{
    cv::Mat Rcv;
    cv::Rodrigues(rvec, Rcv);
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r,c) = Rcv.at<double>(r,c);
    return R;
}

// Predict current pixel for landmark j (mean only). Returns false if outside image.
[[nodiscard]] static bool predictLmPixel(const SystemSLAMPointLandmarks& sys,
                                         const Camera& camera,
                                         std::size_t j,
                                         cv::Point2d& uv_out)
{
    const Eigen::VectorXd xbar = sys.density.mean();
    if (j >= sys.numberLandmarks()) return false;
    const std::size_t idx = sys.landmarkPositionIndex(j);
    if (idx + 2 >= static_cast<std::size_t>(xbar.size())) return false;

    // Mean body pose T_nb → camera pose T_nc
    Pose<double> Tnb;
    Tnb.translationVector = xbar.segment<3>(6);
    Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
    const Pose<double> Tnc = camera.bodyToCamera(Tnb);

    const cv::Vec3d rPNn(xbar(idx+0), xbar(idx+1), xbar(idx+2));
    if (!camera.isWorldWithinFOV(rPNn, Tnc)) return false;

    const cv::Vec2d uv = camera.worldToPixel(rPNn, Tnc);
    uv_out = cv::Point2d(uv[0], uv[1]);
    return camera.isPixelInside(Eigen::Vector2d(uv[0], uv[1]));
}

/*
runVisualNavigationFromVideo:
Main loop orchestrating model (4)–(5) and measurement updates.

Model (shared across scenarios):
  x = [ν(6), r^n_{B/N}(3), Θ^n_B(3), m...] with dη/dt = J_K(η)ν, dm/dt = 0  (4)–(5).

Scenario 1 measurement (unique tags):
  Landmark m_j = [ r^n_{j/N}, Θ^n_j ]^T (6); 3D corners from tag frame (8)–(9);
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
        mu_body.segment<3>(9) << 0.0, 0.0, 0.0;  // Θ^n_B

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12);
        S_body.block<3,3>(0,0) *= 1.0;           // v uncertainty
        S_body.block<3,3>(3,3) *= 0.5;           // ω uncertainty
        const double d2r = (1.0 * std::numbers::pi / 180.0);
        S_body.block<3,3>(6,6) *= 0.1;           // r uncertainty (≈1 cm)
        S_body.block<3,3>(9,9) *= (2 * d2r);     // Θ uncertainty (≈1°)

        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPoseLandmarks>(SystemSLAMPoseLandmarks(p0));
    }
    else if (scenario == 2) {
        // Point-landmark SLAM system
        Eigen::VectorXd mu_body(12);  mu_body.setZero();        // ν(6), r(3), Θ(3)
        mu_body.segment<3>(6) << 0.0, 0.0, -1.0;                // r^n_{B/N}
        mu_body.segment<3>(9) << -std::numbers::pi/2.0, -std::numbers::pi/2.0, 0.0; // Θ^n_B

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12);
        S_body.block<3,3>(0,0) *= 0.3;           // v uncertainty
        S_body.block<3,3>(6,6) *= 0.3;           // position sqrt-cov
        const double d2r = (1.0 * std::numbers::pi / 180.0);
        S_body.block<3,3>(9,9) *= (10.0 * d2r);  // orientation sqrt-cov
        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(SystemSLAMPointLandmarks(p0));

        // ONNX Duck detector (same path as Lab 3)
        const std::filesystem::path onnx_file = "../data/duck_with_postprocessing.onnx";
        assert(std::filesystem::exists(onnx_file) && "[DuckDetector] ONNX model not found at ../src/duck_with_postprocessing.onnx");
        duckDetector = std::make_unique<DuckDetectorONNX>(onnx_file.string());
        assert(duckDetector && "[DuckDetector] construction failed");
    }
    else if (scenario == 3) {
        // Use the same body prior style as your working Scenario 2 for stability
        Eigen::VectorXd mu_body(12);  mu_body.setZero();
        mu_body.segment<3>(6) << 0.0, 0.0, -1.0;
        mu_body.segment<3>(9) << -std::numbers::pi/2.0, -std::numbers::pi/2.0, 0.0;

        Eigen::MatrixXd S_body = Eigen::MatrixXd::Identity(12,12);
        S_body.block<3,3>(0,0) *= 0.3;
        S_body.block<3,3>(6,6) *= 0.3;
        const double d2r = (1.0 * std::numbers::pi / 180.0);
        S_body.block<3,3>(9,9) *= (10.0 * d2r);

        auto p0 = GaussianInfo<double>::fromSqrtMoment(mu_body, S_body);
        systemPtr = std::make_unique<SystemSLAMPointLandmarks>(SystemSLAMPointLandmarks(p0));
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
    
    // Promotion buffer for Scenario 2 (placeholder for future use)
    struct PendingDuck {
        cv::Point2f uv;
        double      area;
        int         hits;       // re-detection count
        int         lastSeen;   // frame index
    };
    std::vector<PendingDuck> pending;

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
            // --- 1. Detect ArUco Tags ---
            std::vector<cv::Vec3d> rvecs, tvecs;
            ArucoDetections dets = detectArUcoPOSE(
                imgin, cv::aruco::DICT_6X6_250, true,
                camera.cameraMatrix, camera.distCoeffs, TAG_SIZE_METERS,
                &rvecs, &tvecs, nullptr, REPROJ_ERR_THRESH_PX, false);

            auto* sysPose = dynamic_cast<SystemSLAMPoseLandmarks*>(&system);
            assert(sysPose && "Scenario 1 expects SystemSLAMPoseLandmarks");

            // --- 2. Camera pose from SLAM state: T_nb → T_nc using T_bc ---
            const Eigen::VectorXd xmean = sysPose->density.mean();
            Pose<double> Tnb;
            Tnb.translationVector = xmean.segment<3>(6);
            Tnb.rotationMatrix    = rpy2rot(xmean.segment<3>(9));
            const Pose<double> Tnc = camera.bodyToCamera(Tnb); // Tnc = Tnb * Tbc

            // --- 3. Initialize new landmarks by unique tag ID ---
            if (id_by_landmark.size() < system.numberLandmarks()) {
                id_by_landmark.resize(system.numberLandmarks(), -1);
            }

            for (std::size_t i = 0; i < dets.ids.size(); ++i)
            {
                const int tagId = dets.ids[i];
                if (id2lm.count(tagId)) continue; // already in map

                // Pose of tag in camera: T_cj
                Pose<double> Tcj(rodriguesToRot(rvecs[i]),
                                 Eigen::Vector3d(tvecs[i][0], tvecs[i][1], tvecs[i][2]));

                // World pose of tag: T_nj = T_nc * T_cj
                Pose<double> Tnj = Tnc * Tcj;
                const Eigen::Vector3d rnj     = Tnj.translationVector;
                const Eigen::Vector3d Thetanj = rot2rpy(Tnj.rotationMatrix);

                // Standard initial uncertainty
                Eigen::Matrix<double,6,6> Sj = Eigen::Matrix<double,6,6>::Identity();
                Sj.diagonal() << INIT_POS_SIGMA_TAG, INIT_POS_SIGMA_TAG, INIT_POS_SIGMA_TAG,
                                  INIT_ANG_SIGMA_TAG, INIT_ANG_SIGMA_TAG, INIT_ANG_SIGMA_TAG;
                
                const std::size_t j = sysPose->appendLandmark(rnj, Thetanj, Sj);

                if (j >= id_by_landmark.size()) id_by_landmark.resize(j+1, -1);
                id_by_landmark[j] = tagId;
                id2lm[tagId]      = j;
            }

            // --- 4. Build measurement and process (4 corners per detected tag) ---
            const std::size_t N = dets.ids.size();
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, 4*N);
            for (std::size_t i = 0; i < N; ++i) {
                for (int k = 0; k < 4; ++k) {
                    Y(0, 4*i + k) = dets.corners[i][k].x;
                    Y(1, 4*i + k) = dets.corners[i][k].y;
                }
            }
            system.view() = dets.annotated.empty() ? imgin : dets.annotated;

            MeasurementSLAMUniqueTagBundle meas(t, Y, camera, dets.ids);
            meas.setIdByLandmark(id_by_landmark);
            meas.process(system);
            id_by_landmark = meas.idByLandmark();

            // --- 5. Visualization ---
            plot.setData(system, meas);
        }
        else if (scenario == 2)
        {
            auto* sysPts = dynamic_cast<SystemSLAMPointLandmarks*>(&system);
            assert(sysPts && "Scenario 2 expects SystemSLAMPointLandmarks");
            assert(duckDetector && "duckDetector must be initialised in scenario 2");

            auto stateDimOK = [&]()->bool {
                const std::size_t nLM = sysPts->numberLandmarks();
                const std::size_t dim = static_cast<std::size_t>(system.density.mean().size());
                return (dim == 12u + 3u*nLM);
            };

            // ============================ Detect ============================
            cv::Mat viz = duckDetector->detect(imgin);
            const auto& C   = duckDetector->last_centroids();
            const auto& Apx = duckDetector->last_areas();
            system.view() = viz;

            if (Apx.size() != C.size()) {
                MeasurementPointBundle measEmpty(t, Eigen::Matrix<double,2,Eigen::Dynamic>(2,0), camera);
                plot.setData(system, measEmpty);
                return;
            }

            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, static_cast<int>(C.size()));
            Eigen::VectorXd Avec(static_cast<int>(C.size()));
            for (int i = 0; i < static_cast<int>(C.size()); ++i) {
                Y(0,i) = C[i].x;  Y(1,i) = C[i].y;
                Avec(i) = std::max(1.0, Apx[i]);
            }

            // ============================ Helpers ============================
            auto buildIdxLandmarks = [&](std::vector<std::size_t>& out){
                out.clear();
                if (!stateDimOK()) return;
                const Eigen::VectorXd xbar = system.density.mean();
                Pose<double> Tnb; Tnb.translationVector = xbar.segment<3>(6);
                Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
                const Pose<double> Tnc = camera.bodyToCamera(Tnb);
                for (std::size_t j = 0; j < sysPts->numberLandmarks(); ++j) {
                    const std::size_t idx = sysPts->landmarkPositionIndex(j);
                    if (idx + 2 >= static_cast<std::size_t>(xbar.size())) continue;
                    const cv::Vec3d rPNn(xbar(idx+0), xbar(idx+1), xbar(idx+2));
                    if (camera.isWorldWithinFOV(rPNn, Tnc)) out.push_back(j);
                }
            };

            auto pickSurplusWithNMS = [&](const std::vector<char>& used)->std::vector<int>{
                std::vector<int> surplus;
                for (int i = 0; i < static_cast<int>(Y.cols()); ++i) if (!used[i]) surplus.push_back(i);
                std::sort(surplus.begin(), surplus.end(),
                          [&](int a, int b){ return Avec(a) > Avec(b); }); // prefer larger area
                std::vector<int> chosen;
                chosen.reserve(MAX_NEW_PER_FR);
                for (int idxFeat : surplus) {
                    bool far = true;
                    for (int kept : chosen) {
                        const double dx = Y(0,idxFeat) - Y(0,kept);
                        const double dy = Y(1,idxFeat) - Y(1,kept);
                        if (dx*dx + dy*dy < static_cast<double>(NMS_MIN_SEP_PX*NMS_MIN_SEP_PX)) { far = false; break; }
                    }
                    if (!far) { continue; }
                    chosen.push_back(idxFeat);
                    if (static_cast<int>(chosen.size()) >= MAX_NEW_PER_FR) break;
                }
                return chosen;
            };

            // Robust seeding with duplicate-block + anisotropic init
            auto appendFromDetectionsRobust = [&](const std::vector<int>& detIdx, int K){
                if (K <= 0) return 0;

                // Camera pose & intrinsics at the mean
                const Eigen::VectorXd xbar = system.density.mean();
                Pose<double> Tnb;
                Tnb.translationVector = xbar.segment<3>(6);
                Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
                const Pose<double> Tnc = camera.bodyToCamera(Tnb);
                const Eigen::Matrix3d Rnc = Tnc.rotationMatrix;
                const Eigen::Vector3d rCNn = Tnc.translationVector;
                const double fx = camera.cameraMatrix.at<double>(0,0);
                const double fy = camera.cameraMatrix.at<double>(1,1);

                int added = 0;
                for (int k = 0; k < static_cast<int>(detIdx.size()) && added < K; ++k) {
                    const int iFeat = detIdx[k];
                    if (iFeat < 0 || iFeat >= static_cast<int>(Y.cols())) continue;

                    const double u  = Y(0,iFeat);
                    const double v  = Y(1,iFeat);
                    const double Ai = std::max(1.0, Avec(iFeat));

                    // Depth from area with clamp
                    const double depth2 = (fx * fy * std::numbers::pi * DUCK_RADIUS_M * DUCK_RADIUS_M) / Ai;
                    if (!(depth2 > 0.0) || !std::isfinite(depth2)) continue;
                    double z = std::sqrt(depth2);
                    if (!std::isfinite(z)) continue;
                    z = std::clamp(z, Z_MIN_SEED, Z_MAX_SEED);

                    // Ray in camera frame → world point
                    const cv::Vec3d dirC = camera.pixelToVector(cv::Vec2d(u,v));
                    const Eigen::Vector3d uPCc(dirC[0], dirC[1], dirC[2]);
                    const Eigen::Vector3d rLNn = Rnc * (z * uPCc) + rCNn;

                    // Reprojection gate (seed must reproject near (u,v))
                    const cv::Vec2d uv_back = camera.worldToPixel(cv::Vec3d(rLNn.x(), rLNn.y(), rLNn.z()), Tnc);
                    const double du = uv_back[0] - u;
                    const double dv = uv_back[1] - v;
                    if (du*du + dv*dv > REPROJ_SEED_THRESH_PX*REPROJ_SEED_THRESH_PX) continue;

                    // Block births near existing landmarks’ predicted pixels
                    bool nearExisting = false;
                    {
                        const cv::Point2d uv_new(u,v);
                        for (std::size_t j = 0; j < sysPts->numberLandmarks(); ++j) {
                            cv::Point2d uv_j;
                            if (!predictLmPixel(*sysPts, camera, j, uv_j)) continue;
                            const double dx = uv_new.x - uv_j.x;
                            const double dy = uv_new.y - uv_j.y;
                            if (dx*dx + dy*dy < EXISTING_SEED_BLOCK_PX*EXISTING_SEED_BLOCK_PX) {
                                nearExisting = true; break;
                            }
                        }
                    }
                    if (nearExisting) continue;

                    // Anisotropic sqrt-covariance (camera frame) rotated to world
                    Eigen::Matrix3d S_c = Eigen::Matrix3d::Zero();
                    S_c(0,0) = SIGMA_PERP_SEED;
                    S_c(1,1) = SIGMA_PERP_SEED;
                    S_c(2,2) = SIGMA_DEPTH_SEED;
                    Eigen::Matrix3d Spos = Rnc * S_c * Rnc.transpose();

                    sysPts->appendLandmark(rLNn, Spos);
                    ++added;
                }
                return added;
            };

            // ---------- Lightweight HSV yellow filter + min area ----------
            std::vector<cv::Point2f> C_filt; C_filt.reserve(C.size());
            std::vector<double>      A_filt; A_filt.reserve(Apx.size());

            auto circle_radius_from_area = [](double Ap)->float {
                return std::sqrt(std::max(1.0, Ap) / CV_PI);
            };

            cv::Mat hsv; 
            cv::cvtColor(imgin, hsv, cv::COLOR_BGR2HSV);

            for (size_t i = 0; i < C.size(); ++i) {
                const cv::Point2f c = C[i];
                const double raw_area = std::max(1.0, Apx[i]);
                if (raw_area < MIN_AREA_PX) continue; // area gate

                const float  r = std::max<float>(6.f, circle_radius_from_area(raw_area));

                int x0 = static_cast<int>(std::round(c.x - r)), y0 = static_cast<int>(std::round(c.y - r));
                int w  = static_cast<int>(std::round(2*r)),     h  = static_cast<int>(std::round(2*r));
                if (w<=0 || h<=0) continue;
                if (x0 < 0) { w += x0; x0 = 0; }
                if (y0 < 0) { h += y0; y0 = 0; }
                if (x0 >= hsv.cols || y0 >= hsv.rows) continue;
                if (x0 + w > hsv.cols) w = hsv.cols - x0;
                if (y0 + h > hsv.rows) h = hsv.rows - y0;
                if (w<=0 || h<=0) continue;

                cv::Rect R(x0,y0,w,h);
                cv::Mat roi = hsv(R);

                int hits=0, tot=0;
                for (int yy=0; yy<roi.rows; ++yy) {
                    const uchar* p = roi.ptr<uchar>(yy);
                    for (int xx=0; xx<roi.cols; ++xx) {
                        const int H = p[3*xx+0], S = p[3*xx+1], V = p[3*xx+2];
                        const float dx = float(R.x+xx) - c.x, dy = float(R.y+yy) - c.y;
                        if (dx*dx + dy*dy > r*r) continue; // circle mask
                        if (S >= S_MIN_YELLOW && V >= V_MIN_YELLOW &&
                            H >= H_MIN_YELLOW && H <= H_MAX_YELLOW) ++hits;
                        ++tot;
                    }
                }
                const double ratio = (tot>0) ? double(hits)/double(tot) : 0.0;
                if (ratio >= MIN_YELLOW_RATIO) { 
                    C_filt.push_back(c); 
                    A_filt.push_back(raw_area); 
                }
            }

            // Build measurement arrays from filtered detections
            Eigen::Matrix<double,2,Eigen::Dynamic> Y2(2, static_cast<int>(C_filt.size()));
            Eigen::VectorXd Avec2(static_cast<int>(C_filt.size()));
            for (int i = 0; i < static_cast<int>(C_filt.size()); ++i) {
                Y2(0,i) = C_filt[i].x;  Y2(1,i) = C_filt[i].y;
                Avec2(i) = std::max(1.0, A_filt[i]);
            }

            // ============================ Bootstrap if empty ============================
            if (sysPts->numberLandmarks() == 0) {
                if (Y2.cols() == 0) {
                    MeasurementPointBundle measEmpty(t, Eigen::Matrix<double,2,Eigen::Dynamic>(2,0), camera);
                    plot.setData(system, measEmpty);
                    return;
                }
                std::vector<char> used0(static_cast<int>(Y2.cols()), 0);
                auto chosen0 = pickSurplusWithNMS(used0);
                int canAdd0 = std::max(0, NMAX_LANDMARKS - static_cast<int>(sysPts->numberLandmarks()));
                appendFromDetectionsRobust(chosen0, std::min(static_cast<int>(chosen0.size()),
                                                             std::min(MAX_NEW_PER_FR, canAdd0)));
                if (sysPts->numberLandmarks() == 0) {
                    MeasurementPointBundle measEmpty(t, Eigen::Matrix<double,2,Eigen::Dynamic>(2,0), camera);
                    plot.setData(system, measEmpty);
                    return;
                }
            }

            // ============================ Pass 1: associate & cull ============================
            std::vector<std::size_t> idxLandmarks;
            buildIdxLandmarks(idxLandmarks);

            if (!idxLandmarks.empty()) {
                MeasurementSLAMDuckBundle meas1(t, Y2, Avec2, camera,
                                                DUCK_RADIUS_M, SIGMA_PX_MEAS, SIGMA_AREA_MEAS);
                meas1.associate(system, idxLandmarks);

                if (stateDimOK()) {
                    const Eigen::VectorXd xbar = system.density.mean();

                    // T_nb → T_nc
                    Pose<double> Tnb;
                    Tnb.translationVector = xbar.segment<3>(6);
                    Tnb.rotationMatrix    = rpy2rot(xbar.segment<3>(9));
                    const Pose<double> Tnc = camera.bodyToCamera(Tnb);

                    const auto& assoc = meas1.idxFeatures();
                    const std::size_t nLM = sysPts->numberLandmarks();
                    for (std::size_t j = 0; j < nLM; ++j) {
                        const std::size_t idx = sysPts->landmarkPositionIndex(j);
                        if (idx + 2 >= static_cast<std::size_t>(xbar.size())) continue;
                        const cv::Vec3d rPNn(xbar(idx+0), xbar(idx+1), xbar(idx+2));
                        const bool predictedVisible = camera.isWorldWithinFOV(rPNn, Tnc);
                        const bool matched = (j < assoc.size() && assoc[j] >= 0);
                        std::ignore = matched;
                        sysPts->updateFailureCounter(j, predictedVisible && !matched);
                    }
                }
                sysPts->cullFailed(CULL_FAIL_THRESH);
            }

            // ============================ Bootstrap during run (surplus) ============================
            {
                std::vector<char> used(static_cast<int>(Y2.cols()), 0);
                std::vector<std::size_t> idxTmp; buildIdxLandmarks(idxTmp);
                if (!idxTmp.empty()) {
                    MeasurementSLAMDuckBundle measMark(t, Y2, Avec2, camera,
                                                       DUCK_RADIUS_M, SIGMA_PX_MEAS, SIGMA_AREA_MEAS);
                    measMark.associate(system, idxTmp);
                    const auto& a = measMark.idxFeatures();
                    for (int j = 0; j < static_cast<int>(a.size()); ++j) {
                        const int fi = a[j];
                        if (fi >= 0 && fi < static_cast<int>(Y2.cols())) used[fi] = 1;
                    }
                }
                auto chosen = pickSurplusWithNMS(used);
                int canAdd = std::max(0, NMAX_LANDMARKS - static_cast<int>(sysPts->numberLandmarks()));
                appendFromDetectionsRobust(chosen, std::min(static_cast<int>(chosen.size()),
                                                            std::min(MAX_NEW_PER_FR, canAdd)));
            }

            // ============================ Final association + update ============================
            std::vector<std::size_t> idxLandmarks2;
            buildIdxLandmarks(idxLandmarks2);
            if (idxLandmarks2.empty()) {
                MeasurementPointBundle measEmpty(t, Eigen::Matrix<double,2,Eigen::Dynamic>(2,0), camera);
                plot.setData(system, measEmpty);
                return;
            }

            MeasurementSLAMDuckBundle meas2(t, Y2, Avec2, camera,
                                            DUCK_RADIUS_M, SIGMA_PX_MEAS, SIGMA_AREA_MEAS);
            meas2.associate(system, idxLandmarks2);
            if (stateDimOK()) {
                meas2.process(system);
            }
            plot.setData(system, meas2);
        }
        else if (scenario == 3)
        {
            auto* sysPts = dynamic_cast<SystemSLAMPointLandmarks*>(&system);
            assert(sysPts && "Scenario 3 expects SystemSLAMPointLandmarks");

            // ---------- 0) Detect & annotate ----------
            std::vector<cv::Point2f> pts;
            cv::Mat annotated = detectAndDrawFeatures(imgin, S3_DETECTOR, S3_FEAT_MAXN, &pts);
            system.view() = annotated;

            // Build measurement Y (2×m)
            const int m = static_cast<int>(pts.size());
            Eigen::Matrix<double,2,Eigen::Dynamic> Y(2, m);
            for (int i = 0; i < m; ++i) { Y(0,i) = pts[i].x; Y(1,i) = pts[i].y; }
            MeasurementPointBundle meas(t, Y, camera);

            // Helper: current camera pose from mean
            auto current_Tnc = [&](){
                const Eigen::VectorXd xbar = system.density.mean();
                Pose<double> Tnb; Tnb.translationVector = xbar.segment<3>(6);
                Tnb.rotationMatrix = rpy2rot(xbar.segment<3>(9));
                return camera.bodyToCamera(Tnb);
            };

            // Shi–Tomasi response image for ranking
            cv::Mat gray, resp;
            if (!imgin.empty()) {
                cv::cvtColor(imgin, gray, cv::COLOR_BGR2GRAY);
                cv::cornerMinEigenVal(gray, resp, 3, 3); // CV_32F
            }
            auto scoreAt = [&](float x, float y)->float {
                if (resp.empty()) return 0.f;
                int xi = std::clamp((int)std::lround(x), 0, resp.cols-1);
                int yi = std::clamp((int)std::lround(y), 0, resp.rows-1);
                return resp.at<float>(yi, xi);
            };

            // ---------- A) Bootstrap if map empty ----------
            if (sysPts && sysPts->numberLandmarks() == 0 && m > 0)
            {
                const Pose<double> Tnc = current_Tnc();
                const Eigen::Matrix3d Rnc = Tnc.rotationMatrix;
                const Eigen::Vector3d rCNn = Tnc.translationVector;

                // Rank all features by score (desc)
                std::vector<std::pair<float,int>> ranked; ranked.reserve(m);
                for (int i = 0; i < m; ++i) ranked.emplace_back(scoreAt(pts[i].x, pts[i].y), i);
                std::sort(ranked.begin(), ranked.end(),
                        [](auto& a, auto& b){ return a.first > b.first; });

                int birthsRemaining = std::min(S3_BOOTSTRAP_BIRTHS,
                                        std::max(0, S3_NMAX_LANDMARKS - (int)sysPts->numberLandmarks()));
                std::vector<int> birthed; birthed.reserve(birthsRemaining);

                for (auto& si : ranked) {
                    if (birthsRemaining <= 0) break;
                    const int i = si.second;

                    // NMS spacing among births
                    bool far = true;
                    for (int j : birthed) {
                        const double dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
                        if (dx*dx + dy*dy < S3_MIN_SEP_PX*S3_MIN_SEP_PX) { far = false; break; }
                    }
                    if (!far) continue;

                    // Reprojection-gated seeding at fixed depth
                    const double u = pts[i].x, v = pts[i].y;
                    const cv::Vec3d dirC = camera.pixelToVector(cv::Vec2d(u,v));
                    const Eigen::Vector3d uPCc(dirC[0], dirC[1], dirC[2]);
                    const Eigen::Vector3d rLNn = Rnc * (S3_DEPTH_GUESS_M * uPCc) + rCNn;
                    const cv::Vec2d uv_back = camera.worldToPixel(cv::Vec3d(rLNn.x(), rLNn.y(), rLNn.z()), Tnc);
                    const double du = uv_back[0] - u, dv = uv_back[1] - v;
                    if (du*du + dv*dv > S3_REPROJ_GATE_PX*S3_REPROJ_GATE_PX) continue;

                    Eigen::Matrix3d S_c = Eigen::Matrix3d::Zero();
                    S_c(0,0)=S3_SIGMA_PERP_M; S_c(1,1)=S3_SIGMA_PERP_M; S_c(2,2)=S3_SIGMA_ALONG_M;
                    Eigen::Matrix3d Spos = Rnc * S_c * Rnc.transpose();
                    sysPts->appendLandmark(rLNn, Spos);

                    birthed.push_back(i);
                    birthsRemaining--;
                    if ((int)sysPts->numberLandmarks() >= S3_NMAX_LANDMARKS) break;
                }
            }

            // ---------- B) Assoc + deletion (visibility-aware) ----------
            std::vector<std::size_t> idxLandmarks;
            if (sysPts && sysPts->numberLandmarks() > 0)
            {
                const Pose<double> Tnc = current_Tnc();
                const Eigen::VectorXd xbar = system.density.mean();

                // Visible-subset prefilter
                for (std::size_t j = 0; j < sysPts->numberLandmarks(); ++j) {
                    const std::size_t base = sysPts->landmarkPositionIndex(j);
                    const cv::Vec3d rPNn(xbar(base+0), xbar(base+1), xbar(base+2));
                    if (camera.isWorldWithinFOV(rPNn, Tnc)) idxLandmarks.push_back(j);
                }

                if (!idxLandmarks.empty()) {
                    meas.associate(system, idxLandmarks);
                    meas.process(system);
                }

                // Deletion: increment failure for predicted-visible but unmatched
                {
                    const auto& assoc = meas.idxFeatures();
                    const std::size_t nLM = sysPts->numberLandmarks();
                    for (std::size_t j = 0; j < nLM; ++j) {
                        const std::size_t base = sysPts->landmarkPositionIndex(j);
                        const cv::Vec3d rPNn(xbar(base+0), xbar(base+1), xbar(base+2));
                        const bool predictedVisible = camera.isWorldWithinFOV(rPNn, Tnc);

                        bool matched = false;
                        if (j < assoc.size()) matched = (assoc[j] >= 0);

                        sysPts->updateFailureCounter(j, predictedVisible && !matched);
                    }
                    sysPts->cullFailed(S3_CULL_FAIL_THRESH);
                }

                // ---------- C) STRICT births per frame (top-score unmatched) ----------
                if ((int)sysPts->numberLandmarks() < S3_NMAX_LANDMARKS && m > 0)
                {
                    // Mark matched features
                    std::vector<char> used(m, 0);
                    const auto& assoc = meas.idxFeatures();
                    for (int j = 0; j < (int)assoc.size(); ++j) {
                        const int fi = assoc[j];
                        if (fi >= 0 && fi < m) used[fi] = 1;
                    }

                    // Rank UNMATCHED by score (desc)
                    std::vector<std::pair<float,int>> ranked; ranked.reserve(m);
                    for (int i = 0; i < m; ++i) if (!used[i])
                        ranked.emplace_back(scoreAt(pts[i].x, pts[i].y), i);
                    std::sort(ranked.begin(), ranked.end(),
                            [](auto& a, auto& b){ return a.first > b.first; });

                    int birthsRemaining = std::min(S3_BIRTHS_PER_FRAME,
                                            std::max(0, S3_NMAX_LANDMARKS - (int)sysPts->numberLandmarks()));
                    std::vector<int> birthed; birthed.reserve(birthsRemaining);

                    const Pose<double> TncB = current_Tnc();
                    const Eigen::Matrix3d Rnc = TncB.rotationMatrix;
                    const Eigen::Vector3d rCNn = TncB.translationVector;

                    for (auto& si : ranked) {
                        if (birthsRemaining <= 0) break;
                        const int i = si.second;

                        // NMS among births (this frame)
                        bool far = true;
                        for (int j : birthed) {
                            const double dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
                            if (dx*dx + dy*dy < S3_MIN_SEP_PX*S3_MIN_SEP_PX) { far = false; break; }
                        }
                        if (!far) continue;

                        // Reprojection gating
                        const double u = pts[i].x, v = pts[i].y;
                        const cv::Vec3d dirC = camera.pixelToVector(cv::Vec2d(u,v));
                        const Eigen::Vector3d uPCc(dirC[0], dirC[1], dirC[2]);
                        const Eigen::Vector3d rLNn = Rnc * (S3_DEPTH_GUESS_M * uPCc) + rCNn;
                        const cv::Vec2d uv_back = camera.worldToPixel(cv::Vec3d(rLNn.x(), rLNn.y(), rLNn.z()), TncB);
                        const double du = uv_back[0] - u, dv = uv_back[1] - v;
                        if (du*du + dv*dv > S3_REPROJ_GATE_PX*S3_REPROJ_GATE_PX) continue;

                        // Append newborn
                        Eigen::Matrix3d S_c = Eigen::Matrix3d::Zero();
                        S_c(0,0)=S3_SIGMA_PERP_M; S_c(1,1)=S3_SIGMA_PERP_M; S_c(2,2)=S3_SIGMA_ALONG_M;
                        Eigen::Matrix3d Spos = Rnc * S_c * Rnc.transpose();
                        sysPts->appendLandmark(rLNn, Spos);

                        birthed.push_back(i);
                        birthsRemaining--;
                        if ((int)sysPts->numberLandmarks() >= S3_NMAX_LANDMARKS) break;
                    }

                    // Newborn-only association so they can turn blue immediately
                    if (!birthed.empty()) {
                        const int newCount = (int)sysPts->numberLandmarks();
                        const int oldCount = newCount - (int)birthed.size();
                        std::vector<std::size_t> idxNew; idxNew.reserve(birthed.size());
                        for (int j = oldCount; j < newCount; ++j) idxNew.push_back((std::size_t)j);
                        if (!idxNew.empty()) {
                            meas.associate(system, idxNew);
                            meas.process(system);
                        }
                    }
                }

                // ---------- D) MERGE CLOSE LANDMARKS (deduplicate) ----------
                // Strategy: if two landmarks are close in 3D AND their predicted pixels are close,
                // keep the one with smaller positional uncertainty (trace of sqrt-cov),
                // force-delete the other by bumping its failure counter to the cull threshold,
                // then call cullFailed() once.
                {
                    const int nLM = (int)sysPts->numberLandmarks();
                    const Pose<double> TncM = current_Tnc();

                    auto predictPix = [&](int j, cv::Point2d& uv)->bool {
                        return predictLmPixel(*sysPts, camera, (std::size_t)j, uv);
                    };

                    // Collect pairs to merge (limit work with a per-frame cap)
                    int mergesDone = 0;
                    std::vector<char> killed(nLM, 0);

                    // Pre-compute means & sqrt-cov traces for speed
                    std::vector<Eigen::Vector3d> mu(nLM);
                    std::vector<double> trS(nLM, 0.0);
                    for (int j = 0; j < nLM; ++j) {
                        const GaussianInfo<double> pd = sysPts->landmarkPositionDensity(j);
                        mu[j] = pd.mean();
                        const Eigen::Matrix3d Sc = pd.sqrtCov();
                        trS[j] = Sc(0,0)*Sc(0,0) + Sc(1,1)*Sc(1,1) + Sc(2,2)*Sc(2,2); // proxy for uncertainty
                    }

                    for (int a = 0; a < nLM && mergesDone < S3_MERGE_MAX_PER_FRAME; ++a) {
                        if (killed[a]) continue;
                        cv::Point2d uva;
                        const bool a_has_uv = predictPix(a, uva);

                        for (int b = a+1; b < nLM && mergesDone < S3_MERGE_MAX_PER_FRAME; ++b) {
                            if (killed[b]) continue;

                            // 3D distance check
                            const double d3 = (mu[a] - mu[b]).norm();
                            if (d3 > S3_MERGE_DIST_M) continue;

                            // Optional pixel distance check (both must project)
                            cv::Point2d uvb;
                            if (!a_has_uv || !predictPix(b, uvb)) continue;
                            const double dx = uva.x - uvb.x, dy = uva.y - uvb.y;
                            if (dx*dx + dy*dy > S3_MERGE_PIX_PX*S3_MERGE_PIX_PX) continue;

                            // Choose survivor: smaller positional uncertainty wins
                            int keep = a, kill = b;
                            if (trS[b] < trS[a]) { keep = b; kill = a; }

                            // Force-delete 'kill' by bumping its failure counter to threshold
                            for (int ttt = 0; ttt < S3_CULL_FAIL_THRESH; ++ttt) {
                                sysPts->updateFailureCounter((std::size_t)kill, true);
                            }
                            killed[kill] = 1;
                            mergesDone++;
                        }
                    }

                    if (mergesDone > 0) {
                        sysPts->cullFailed(S3_CULL_FAIL_THRESH);
                    }
                }
            }

            // ---------- E) Plot ----------
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
