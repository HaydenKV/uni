#include <print>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include "to_string.hpp"
#include "BufferedVideo.h"
#include "Pose.hpp"
#include "rotation.hpp"
#include "Camera.h"
#include "DJIVideoCaption.h"
#include "GaussianInfo.hpp"
#include "funcmin.hpp"
#include "SystemVisualNav.h"
#include "MeasurementOutdoorFlowBundle.h"
#include "visual_odometry.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// Forward declarations
static Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0);
static void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor);

void runVisualOdometryFromVideo(const std::filesystem::path & videoPath,
                                const std::filesystem::path & cameraPath,
                                const std::filesystem::path & outputDirectory)
{
    // 1: Lab 11 — sampling & plotting scale
    int imgModulus  = 10;   // Take frames divisible by this number
    int divisor     = 2;    // Image scaling factor (used for plotting only)

    assert(!videoPath.empty());

    // Subtitle path
    std::filesystem::path subtitlePath = videoPath.parent_path() / (videoPath.stem().string() + ".SRT");
    assert(std::filesystem::exists(subtitlePath));

    // Load and parse subtitle file
    std::vector<DJIVideoCaption> djiVideoCaption = getVideoCaptions(subtitlePath);

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_"
                                   + std::to_string(divisor)
                                   + "_"
                                   + std::to_string(imgModulus)
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
    Camera camera;
    assert(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> camera;

    // Display loaded calibration data
    camera.printCalibration();

    // Set camera pose w.r.t. body (T^b_c)
    // Lab spec: b1 = c3, b2 = c1, b3 = c2  (columns of Rbc are body axes expressed in camera frame)
    Eigen::Matrix3d Rbc_nom;
    Rbc_nom << 0, 0, 1,
               1, 0, 0,
               0, 1, 0;

    // (Optional) small gimbal biases to match real tilt
    double cam_pitch_deg = 1.6;   // positive pitches camera nose down
    double cam_roll_deg  = -0.3;  // small roll
    double cam_pitch_rad = cam_pitch_deg * M_PI / 180.0;
    double cam_roll_rad  = cam_roll_deg  * M_PI / 180.0;

    Eigen::Matrix3d Rbc = Rbc_nom * rotx(cam_pitch_rad) * rotz(cam_roll_rad);
    const Eigen::Vector3d tbc = Eigen::Vector3d::Zero();
    camera.Tbc = Pose<double>(Rbc, tbc);

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);

    std::println("Input video: {}", videoPath.string());
    std::println("Subtitle file: {}", subtitlePath.string());
    std::println("Total number of frames: {}", nFrames);
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::println("Input video frame rate: {}", fps);
    std::println("Input video dimensions: [{} x {}]",
                 cap.get(cv::CAP_PROP_FRAME_WIDTH),
                 cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
        frameSize.height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        double outputFps = fps/imgModulus;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // mp4v
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual odometry (dummy system, we only use measurement cost)
    auto p0 = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), Eigen::MatrixXd::Zero(18, 18));
    SystemVisualNav system(p0);
    Eigen::VectorXd etakm1(6);
    Eigen::VectorXd etak(6);
    etak = getInitialPose(djiVideoCaption[0]);

    cv::Mat imgk_raw;
    cv::Mat imgkm1_raw;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1;

    for (int i = 0, k = 0;; ++i)
    {
        // Capture frame by frame
        imgk_raw = bufferedVideoReader.read();
        if (imgk_raw.empty())
        {
            break;
        }

        if (i % imgModulus == 0)
        {
            if (k > 0)
            {
                // Create measurement data from image pair and previous tracked features
                MeasurementOutdoorFlowBundle measurement(i/fps, camera, imgk_raw, imgkm1_raw, rQOikm1);

                rQOikm1 = measurement.trackedPreviousFeatures();
                const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik = measurement.trackedCurrentFeatures();

                // Create cost function with prototype V = costFunc(eta, g, H)
                auto costFunc = [&](const Eigen::VectorXd & etak, Eigen::VectorXd & g, Eigen::MatrixXd & H)
                {
                    return measurement.costOdometry(etak, etakm1, g, H);
                };

                // Minimise cost (maximise log likelihood)
                etak = etakm1; // Start optimisation at previous pose
                const int verbosity = 3; // 0:none, 1:dots, 2:summary, 3:iter
                int ret = funcmin::NewtonTrust(costFunc, etak, verbosity);
                assert(ret == 0);

                Eigen::Vector3d rBNn = etak.head<3>();
                Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());

                std::print("rBNn: \n{}\n", to_string(rBNn));
                std::print("Rnb: \n{}\n", to_string(Rnb));

                Eigen::VectorXd x(18);
                x.setZero();
                x.segment<6>(6)  = etak;
                x.segment<6>(12) = etakm1;

                // Prepare output image
                cv::Mat imgout;
                cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor);

                // Predicted flow field (used for plotting onto original image)
                Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = measurement.predictedFeatures(x, system);

                // Plotting
                std::vector<cv::Point2d> rQOikm1_scaled, rQOik_scaled, rQOik_hat_scaled;
                int np = rQOik.cols();
                rQOikm1_scaled.resize(np);
                rQOik_scaled.resize(np);
                rQOik_hat_scaled.resize(np);
                for (int j = 0; j < np; ++j)
                {
                    rQOikm1_scaled[j].x = rQOikm1(0, j)/divisor;
                    rQOikm1_scaled[j].y = rQOikm1(1, j)/divisor;

                    rQOik_scaled[j].x   = rQOik(0, j)/divisor;
                    rQOik_scaled[j].y   = rQOik(1, j)/divisor;

                    rQOik_hat_scaled[j].x = rQOik_hat(0, j)/divisor;
                    rQOik_hat_scaled[j].y = rQOik_hat(1, j)/divisor;
                }

                // Plot flow vectors (predicted=blue, inlier=green, outlier=red)
                for (int j = 0; j < rQOik.cols(); ++j)
                {
                    cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_hat_scaled[j],
                                    cv::Scalar(255, 0, 0), 2, cv::LINE_AA, 0, 0.25);

                    if (measurement.inlierMask()[j])
                    {
                        cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j],
                                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0, 0.25);
                    }
                    else
                    {
                        cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j],
                                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.25);
                    }
                }

                plotGroundPlane(imgout, etak, camera, divisor);
                plotHorizon(imgout, etak, camera, divisor);
                plotCompass(imgout, etak, camera, divisor);
                plotEpipole(imgout, etak, etakm1, camera, divisor);

                cv::imshow("Visual odometry demo", imgout);
                char key = cv::waitKey(1);
                if (key == 'q')
                {
                    std::println("Key '{}' pressed. Terminating program.", key);
                    break;
                }

                if (doExport)
                {
                    bufferedVideoWriter.write(imgout);
                }

                rQOikm1.resize(2, rQOik.cols());
                rQOikm1 = rQOik;
            }

            imgk_raw.copyTo(imgkm1_raw);
            etakm1 = etak;
            k++;
        }
    }

    if (doExport)
    {
        bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();
}

Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
{
    double h  = caption0.altitude;   // Altitude (GPS) [m]
    double ga = h - 7.0;             // Altitude (AGL) [m]

    Eigen::Vector6d eta0;
    // 3: Lab 11 — NED: D positive down; being above the ground means D = -AGL
    eta0 << 0.0, 0.0, -ga,   0.0, 0.0, 0.0;
    return eta0;
}

void plotGroundPlane(cv::Mat & img,
                     const Eigen::Vector6d & etak,
                     const Camera & camera,
                     const int & divisor)
{
    // ---------------- Tuning (yours) ----------------
    const double Dplane        = 0.0;      // Ground plane (Down=0)
    const double span          = 1000.0;   // World extent (±span) in metres
    const double spacing       = 100.0;    // Grid spacing (metres)
    const double segmentLen    = 10.0;     // Segment length along each grid line (metres)
    const cv::Scalar gridCol   = cv::Scalar(0,0,0); // black
    const double thickness     = 1.0;      // rounded to int for OpenCV
    const double thicknessAxis = 1.0;      // rounded to int (N=0 / E=0)
    const int    lineType      = cv::LINE_AA;
    const double N0            = 0.0;      // centre grid
    const double E0            = 0.0;

    const int th  = std::max<int>(1, std::lround(thickness));
    const int tha = std::max<int>(1, std::lround(thicknessAxis));

    // Slight border inflate so lines reach the edges after rounding/clipping
    const cv::Rect roi(-1, -1, img.cols + 2, img.rows + 2);

    // Pose T^n_b (NED)
    const Eigen::Vector3d rBNn = etak.head<3>();
    const Eigen::Matrix3d Rnb  = rpy2rot(etak.tail<3>());
    Pose<double> Tnb(Rnb, rBNn);

    // Compose world->camera once using Camera::Tbc
    const Pose<double> Tnc      = Tnb * camera.Tbc;
    const Eigen::Matrix3d R_WC  = Tnc.rotationMatrix;     // world-from-camera
    const Eigen::Vector3d r_WC  = Tnc.translationVector;  // camera centre in world

    // Quick depth test (culls the "second plane" behind camera)
    auto camDepthZ = [&](double N, double E)->double {
        const Eigen::Vector3d PW(N, E, Dplane);
        const Eigen::Vector3d PC = R_WC.transpose() * (PW - r_WC);
        return PC.z();
    };

    // Endpoint FOV predicate using Camera's helper (fast)
    auto inFOV = [&](double N, double E)->bool {
        return camera.isWorldWithinFOV(cv::Vec3d(N, E, Dplane), Tnb);
    };

    // Project world -> downscaled pixels (requires z>0 and finite projection)
    auto projectFront = [&](double N, double E, cv::Point& p)->bool
    {
        if (camDepthZ(N, E) <= 0.0) return false;
        cv::Vec2d uv = camera.worldToPixel(cv::Vec3d(N, E, Dplane), Tnb);
        if (!std::isfinite(uv[0]) || !std::isfinite(uv[1])) return false;
        p.x = cvRound(uv[0] / divisor);
        p.y = cvRound(uv[1] / divisor);
        return true;
    };

    // Draw one polyline family: either N=const (sweep E) or E=const (sweep N)
    auto drawFamily = [&](bool Nconst, double fixedVal, int thicknessPixels)
    {
        auto makePoint = [&](double s)->cv::Vec3d {
            return Nconst ? cv::Vec3d(fixedVal, s, Dplane)   // N fixed, sweep E
                          : cv::Vec3d(s, fixedVal, Dplane);   // E fixed, sweep N
        };
        auto fovCheck = [&](double s)->bool {
            if (Nconst) return inFOV(fixedVal, s);
            else        return inFOV(s, fixedVal);
        };
        auto project  = [&](double s, cv::Point& p)->bool {
            if (Nconst) return projectFront(fixedVal, s, p);
            else        return projectFront(s, fixedVal, p);
        };

        const double sBegin = (Nconst ? (E0 - span) : (N0 - span));
        const double sEnd   = (Nconst ? (E0 + span) : (N0 + span));

        // March along the line in fixed world steps; segment-wise FOV gate
        for (double s0 = sBegin; s0 < sEnd; s0 += segmentLen)
        {
            const double s1 = std::min(s0 + segmentLen, sEnd);

            // *** Segment FOV test: draw only if at least ONE endpoint is inside ***
            const bool f0 = fovCheck(s0);
            const bool f1 = fovCheck(s1);
            if (!(f0 || f1)) continue;  // fast reject: whole segment outside FOV

            // Project endpoints (require z>0 for both to avoid behind-camera artefacts)
            cv::Point p0, p1;
            const bool ok0 = project(s0, p0);
            const bool ok1 = project(s1, p1);

            if (ok0 && ok1)
            {
                // Clip to the image and draw
                cv::Point a = p0, b = p1;
                if (cv::clipLine(roi, a, b))
                    cv::line(img, a, b, gridCol, thicknessPixels, lineType);
            }
            else
            {
                // Optional: light salvage near borders — sample midpoint once
                // (kept minimal for performance; remove this block if you want maximum speed)
                const double sm = 0.5 * (s0 + s1);
                cv::Point pm;
                if (fovCheck(sm) && project(sm, pm))
                {
                    if (ok0) {
                        cv::Point a = p0, b = pm;
                        if (cv::clipLine(roi, a, b))
                            cv::line(img, a, b, gridCol, thicknessPixels, lineType);
                    }
                    if (ok1) {
                        cv::Point a = pm, b = p1;
                        if (cv::clipLine(roi, a, b))
                            cv::line(img, a, b, gridCol, thicknessPixels, lineType);
                    }
                }
            }
        }
    };

    // ----- N-constant lines (sweep E) -----
    const double NStart = std::floor((N0 - span)/spacing) * spacing;
    const double NEnd   = std::floor((N0 + span)/spacing) * spacing;
    for (double N = NStart; N <= NEnd + 1e-9; N += spacing)
    {
        const int t = (std::abs(N) < 1e-9) ? tha : th;
        drawFamily(/*Nconst=*/true, N, t);
    }

    // ----- E-constant lines (sweep N) -----
    const double EStart = std::floor((E0 - span)/spacing) * spacing;
    const double EEnd   = std::floor((E0 + span)/spacing) * spacing;
    for (double E = EStart; E <= EEnd + 1e-9; E += spacing)
    {
        const int t = (std::abs(E) < 1e-9) ? tha : th;
        drawFamily(/*Nconst=*/false, E, t);
    }
}

void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    // 5: Lab 11 — horizon line via l ~ K^{-T} R_cw n
    Eigen::Matrix3d K; cv::cv2eigen(camera.cameraMatrix, K);
    const Eigen::Matrix3d KinvT = K.inverse().transpose();

    // Rotations: R^n_c = R^n_b R^b_c ; R^c_w = (R^n_c)^T
    const Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;
    const Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
    const Eigen::Matrix3d Rnc = Rnb * Rbc;
    const Eigen::Matrix3d Rcw = Rnc.transpose();

    // Ground normal in world (NED): +D axis
    const Eigen::Vector3d nw(0,0,1);

    // Horizon line in pixel coords (unnormalised)
    const Eigen::Vector3d l = KinvT * (Rcw * nw);  // [a,b,c]^T

    const double a = l.x(), b = l.y(), c = l.z();
    const int Wfull = img.cols * divisor;
    const int Hfull = img.rows * divisor;

    // Intersections with image borders
    std::vector<cv::Point2d> pts;
    auto add = [&](double x, double y){
        if (x >= 0 && x <= Wfull-1 && y >= 0 && y <= Hfull-1)
            pts.emplace_back(x / divisor, y / divisor);
    };

    if (std::abs(b) > 1e-12) {
        add(0,         -c / b);
        add(Wfull-1.0, -(a*(Wfull-1.0) + c)/b);
    }
    if (std::abs(a) > 1e-12) {
        add(-c / a,                0);
        add(-(b*(Hfull-1.0)+c)/a,  Hfull-1.0);
    }

    if (pts.size() >= 2) {
        cv::line(img, pts[0], pts[1], cv::Scalar(0,0,255), 2, cv::LINE_AA);
    }
}

void plotCompass(cv::Mat & img,
                 const Eigen::Vector6d & etak,
                 const Camera & camera,
                 const int & divisor)
{
    // Rotations & intrinsics
    const Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;     // body->cam
    const Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());        // body from NED
    const Eigen::Matrix3d Rnc = Rnb * Rbc;                      // cam from NED
    const Eigen::Matrix3d Rcw = Rnc.transpose();                // world(NED)->cam

    Eigen::Matrix3d K; cv::cv2eigen(camera.cameraMatrix, K);

    const int W = img.cols, H = img.rows;
    auto inside = [&](double x, double y){ return (x>=0 && x<W && y>=0 && y<H); };

    // Distortion-aware projection of a horizontal direction at azimuth (deg)
    auto projDeg = [&](double deg, Eigen::Vector3d& d_c, cv::Point2d& p)->bool {
        const double a = deg * M_PI / 180.0;
        Eigen::Vector3d d_w(std::cos(a), std::sin(a), 0.0);    // NED: N=+x, E=+y
        d_c = Rcw * d_w;                                       // camera ray
        if (!std::isfinite(d_c.z()) || d_c.z() <= 1e-9) return false;   // must face camera
        Eigen::Vector2d pix = camera.vectorToPixel(d_c);       // full camera model (with distortion)
        p.x = pix.x()/divisor; p.y = pix.y()/divisor;
        return std::isfinite(p.x) && std::isfinite(p.y) && inside(p.x, p.y);
    };

    // Estimate horizon tangent at azimuth using small angular step
    auto tangentAtDeg = [&](double deg)->cv::Point2d {
        constexpr double dth = 2.0; // degrees
        Eigen::Vector3d dc0, dc1, dc2; cv::Point2d p0, p1, p2;
        bool ok0 = projDeg(deg,      dc0, p0);
        bool ok1 = projDeg(deg+dth,  dc1, p1);
        bool ok2 = projDeg(deg-dth,  dc2, p2);
        cv::Point2d t(1.0, 0.0);
        if (ok1 && ok2)      { t.x = p1.x - p2.x; t.y = p1.y - p2.y; }
        else if (ok1 && ok0) { t.x = p1.x - p0.x; t.y = p1.y - p0.y; }
        else if (ok0 && ok2) { t.x = p0.x - p2.x; t.y = p0.y - p2.y; }
        double n = std::hypot(t.x, t.y);
        if (n > 1e-6) { t.x /= n; t.y /= n; }
        return t;
    };

    auto drawBlackText = [&](cv::Mat& im, const std::string& txt, cv::Point org,
                             int fontFace, double fontScale, int thickness)
    {
        cv::putText(im, txt, org, fontFace, fontScale, cv::Scalar(0,0,0), thickness, cv::LINE_AA);
    };

    struct Lpair { const char* A; double degA; const char* B; double degB; };
    const std::array<Lpair,4> pairs = {{
        {"N",   0.0,  "S", 180.0},
        {"E",  90.0,  "W", 270.0},
        {"NE", 45.0,  "SW",225.0},
        {"SE",135.0,  "NW",315.0}
    }};

    // Font & layout
    const int fontFace       = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale   = 1.0;
    const int textThickness  = 1;
    const int n_off_px       = -50;  // lift label above the horizon curve

    for (const auto& P : pairs) {
        // Try both directions, choose the visible one (and the one more in front if both)
        Eigen::Vector3d dcA, dcB; cv::Point2d pA, pB;
        const bool okA = projDeg(P.degA, dcA, pA);
        const bool okB = projDeg(P.degB, dcB, pB);

        const char* name = nullptr; double useDeg = 0.0; cv::Point2d p;
        if (okA && !okB) { name = P.A; useDeg = P.degA; p = pA; }
        else if (!okA && okB) { name = P.B; useDeg = P.degB; p = pB; }
        else if (okA && okB) {
            name = (dcA.z() >= dcB.z()) ? P.A : P.B;
            useDeg = (name==P.A) ? P.degA : P.degB;
            p      = (name==P.A) ? pA : pB;
        } else {
            continue; // neither visible
        }

        // Tangent & normal at that azimuth for offset position
        const cv::Point2d t = tangentAtDeg(useDeg);
        const cv::Point2d n(-t.y, t.x);

        // Label position (centered on text, lifted along normal)
        int base=0; cv::Size sz = cv::getTextSize(name, fontFace, fontScale, textThickness, &base);
        cv::Point pos(cvRound(p.x - sz.width*0.5 + n_off_px*n.x),
                      cvRound(p.y + sz.height*0.5 + n_off_px*n.y));

        // Leader line from horizon point up to near the label
        cv::Point p_up(cvRound(p.x + n.x*(n_off_px-2)), cvRound(p.y + n.y*(n_off_px-2)));
        cv::line(img, p, p_up, cv::Scalar(0,0,0), 1, cv::LINE_AA);

        // Black text label
        drawBlackText(img, name, pos, fontFace, fontScale, textThickness);
    }
}

void plotEpipole(cv::Mat & img,
                 const Eigen::Vector6d & etak,
                 const Eigen::Vector6d & etakm1,
                 const Camera & camera,
                 const int & divisor)
{
    // Intrinsics
    Eigen::Matrix3d K; cv::cv2eigen(camera.cameraMatrix, K);

    // Current rotations
    const Eigen::Matrix3d Rbc   = camera.Tbc.rotationMatrix;
    const Eigen::Matrix3d Rnb_k = rpy2rot(etak.tail<3>());
    const Eigen::Matrix3d Rnc_k = Rnb_k * Rbc;
    const Eigen::Matrix3d Rcw_k = Rnc_k.transpose();

    // Camera centers in world
    const Eigen::Vector3d Ck   = etak.head<3>();
    const Eigen::Vector3d Ckm1 = etakm1.head<3>();

    // Direction to previous center expressed in current camera
    const Eigen::Vector3d t_k = Rcw_k * (Ckm1 - Ck);

    // Epipole (homogeneous)
    const Eigen::Vector3d e = K * t_k;
    if (!std::isfinite(e.z()) || std::abs(e.z()) < 1e-9) return;

    const cv::Point2d pe((e.x()/e.z())/divisor, (e.y()/e.z())/divisor);

    // Draw a larger hollow orange circle
    const cv::Scalar ORANGE(0,165,255);
    const int radius    = 14;
    const int thickness = 3;
    cv::circle(img, pe, radius, ORANGE, thickness, cv::LINE_AA);
}
