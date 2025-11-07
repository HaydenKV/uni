// visual_odometry.cpp (Set 1 behaviour restored, Set 2 flow & plotting retained)

#include <print>
#include <cmath>
#include <optional>
#include <algorithm>
#include <array>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/SVD>
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

// ---- fwd decls ----
static Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0);
static void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor);
static void plotTelemetry(cv::Mat& img, const Eigen::Vector6d& etak, const std::optional<double>& alt_agl, const int& divisor);

// ---- helpers ----
static inline Eigen::Matrix3d Rcw_from(const Eigen::Vector6d& eta, const Camera& cam) {
    const Eigen::Matrix3d Rnb = rpy2rot(eta.tail<3>());
    const Eigen::Matrix3d Rbc = cam.Tbc.rotationMatrix;
    return (Rnb * Rbc).transpose(); // world→camera
}

static std::optional<std::pair<double,int>>
altAGLPastIdx(const std::vector<DJIVideoCaption>& caps, double t, double ground_bias_m=7.0)
{
    if (caps.empty()) return std::nullopt;
    int i_last = -1;
    for (int i = 0; i < (int)caps.size(); ++i) {
        if (caps[i].time <= t) i_last = i; else break;
    }
    if (i_last < 0) return std::nullopt;
    if (!std::isfinite(caps[i_last].altitude)) return std::nullopt;
    return std::make_pair(caps[i_last].altitude - ground_bias_m, i_last);
}

void runVisualOdometryFromVideo(const std::filesystem::path & videoPath,
                                const std::filesystem::path & cameraPath,
                                const std::filesystem::path & outputDirectory)
{
    // ---- Set 2 sampling/plot scale ----
    int imgModulus  = 10;   // process every 10th frame
    int divisor     = 2;    // downscale for display/export

    assert(!videoPath.empty());
    std::filesystem::path subtitlePath = videoPath.parent_path() / (videoPath.stem().string() + ".SRT");
    assert(std::filesystem::exists(subtitlePath));
    std::vector<DJIVideoCaption> djiVideoCaption = getVideoCaptions(subtitlePath);

    // Export
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_"
                                   + std::to_string(divisor)
                                   + "_"
                                   + std::to_string(imgModulus)
                                   + "_PerFeature.mp4";
        outputPath = outputDirectory / outputFilename;
    }

    // Camera
    Camera camera;
    assert(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> camera;
    camera.printCalibration();

    // T^b_c with Set 2 style biases
    Eigen::Matrix3d Rbc_nom;
    Rbc_nom << 0, 0, 1,
               1, 0, 0,
               0, 1, 0;
    const double cam_pitch_deg = 1.6;
    const double cam_roll_deg  = -0.3;
    const double cam_yaw_deg   = 0.0;
    Eigen::Matrix3d Rbc = Rbc_nom * rotx(cam_pitch_deg*M_PI/180.0)
                                   * rotz(cam_roll_deg *M_PI/180.0)
                                   * roty(cam_yaw_deg  *M_PI/180.0);
    camera.Tbc = Pose<>(Rbc, Eigen::Vector3d::Zero());

    // Video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    const int nFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    const double fps  = cap.get(cv::CAP_PROP_FPS);
    std::println("Input video: {}", videoPath.string());
    std::println("Subtitle file: {}", subtitlePath.string());
    std::println("Total frames: {}", nFrames);
    std::println("FPS: {}", fps);
    std::println("Dimensions: [{} x {}]",
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
        const double outputFps = fps / imgModulus;
        int codec = cv::VideoWriter::fourcc('m','p','4','v');
        if (videoOut.open(outputPath.string(), codec, outputFps, frameSize)) {
            bufferedVideoWriter.start(videoOut);
            std::println("Export visualisation to {}", outputPath.string());
        } else {
            std::println("Warning: failed to open writer '{}'. Export disabled.", outputPath.string());
            doExport = false;
        }
    }

    // System
    auto p0 = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), Eigen::MatrixXd::Zero(18, 18));
    SystemVisualNav system(p0);

    // State
    Eigen::VectorXd etakm1(6), etak(6);
    etak = getInitialPose(djiVideoCaption[0]);

    // Frames
    cv::Mat imgk_raw, imgkm1_raw;

    // Tracker seed
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQ_prev_px; // features at (k-1) to seed (k)

    // Alt event state
    static int    last_alt_idx  = -1;
    static double last_alt_agl  = std::numeric_limits<double>::quiet_NaN();

    for (int i = 0, k = 0;; ++i)
    {
        imgk_raw = bufferedVideoReader.read();
        if (imgk_raw.empty()) break;

        if (i % imgModulus != 0) continue; // sample

        if (k > 0)
        {
            // --------- Set 2 optical flow generation (k-1→k) ---------
            MeasurementOutdoorFlowBundle meas2(i/fps, camera, imgk_raw, imgkm1_raw, rQ_prev_px);
            const Eigen::Matrix<double,2,Eigen::Dynamic>& rQ_curr_px = meas2.trackedCurrentFeatures();
            rQ_prev_px = meas2.trackedPreviousFeatures();
            const int np_full = (int)rQ_curr_px.cols();

            // --------- Set 1 horizon-aware filtering + sky rotation prior ---------
            cv::Mat Kcv = camera.cameraMatrix, Dcv = camera.distCoeffs;

            // undistort helper
            auto undist = [&](const Eigen::Matrix<double,2,Eigen::Dynamic>& Ppx)
                -> Eigen::Matrix<double,2,Eigen::Dynamic>
            {
                std::vector<cv::Point2f> in(Ppx.cols()), out;
                for (int j = 0; j < Ppx.cols(); ++j)
                    in[j] = cv::Point2f((float)Ppx(0,j), (float)Ppx(1,j));
                cv::undistortPoints(in, out, Kcv, Dcv);
                Eigen::Matrix<double,2,Eigen::Dynamic> N(2, (int)out.size());
                for (int j = 0; j < (int)out.size(); ++j) { N(0,j) = out[j].x; N(1,j) = out[j].y; }
                return N;
            };

            // normalised copies
            Eigen::Matrix<double,2,Eigen::Dynamic> rQ_prev_n = undist(rQ_prev_px);
            Eigen::Matrix<double,2,Eigen::Dynamic> rQ_curr_n = undist(rQ_curr_px);

            // horizon line in normalised plane using etakm1
            const Eigen::Matrix3d Rcw_n = Rcw_from(etakm1, camera);
            const Eigen::Vector3d n_w(0,0,-1); // NED up
            const Eigen::Vector3d l_n = Rcw_n * n_w; // (a,b,c)·[x_n,y_n,1]=0

            auto sdist_n = [&](double xn, double yn){
                const double a=l_n.x(), b=l_n.y(), c=l_n.z();
                const double den = std::sqrt(a*a + b*b) + 1e-12;
                return (a*xn + b*yn + c) / den;
            };

            // ground sign via bottom-centre pixel
            const double cx_px = Kcv.at<double>(0,2);
            const double Hfull = (double)imgk_raw.rows;
            std::vector<cv::Point2f> bc_in(1), bc_out;
            bc_in[0] = cv::Point2f((float)cx_px, (float)(Hfull-1.0));
            cv::undistortPoints(bc_in, bc_out, Kcv, Dcv);
            const double ground_sign = (sdist_n(bc_out[0].x, bc_out[0].y) >= 0.0) ? 1.0 : -1.0;

            // thresholds in normalised units (~40 px / fx)
            const double fx = Kcv.at<double>(0,0);
            const double BAND_G = 40.0 / fx;
            const double BAND_S = 40.0 / fx;

            // keep list: confident ground or confident sky
            std::vector<int> keep; keep.reserve(np_full);
            for (int j = 0; j < np_full; ++j) {
                const double s_ref  = ground_sign * sdist_n(rQ_prev_n(0,j), rQ_prev_n(1,j));
                const double s_curr = ground_sign * sdist_n(rQ_curr_n(0,j), rQ_curr_n(1,j));
                const bool is_ground = (s_ref  > +BAND_G) && (s_curr > +BAND_G);
                const bool is_sky    = (s_ref  < -BAND_S) && (s_curr < -BAND_S);
                if (is_ground || is_sky) keep.push_back(j);
            }

            // compress to filtered sets; soft-fallback if too small
            Eigen::Matrix<double,2,Eigen::Dynamic> rQ_prev_px_f(2, np_full), rQ_curr_px_f(2, np_full);
            int w = 0;
            for (int j : keep) {
                rQ_prev_px_f.col(w) = rQ_prev_px.col(j);
                rQ_curr_px_f.col(w) = rQ_curr_px.col(j);
                ++w;
            }
            rQ_prev_px_f.conservativeResize(Eigen::NoChange, w);
            rQ_curr_px_f.conservativeResize(Eigen::NoChange, w);

            const int MIN_FEAT = 90;
            const double MIN_FRAC = 0.22;
            const bool enough = (w >= std::max(MIN_FEAT, (int)std::round(MIN_FRAC*np_full)));
            const Eigen::Matrix<double,2,Eigen::Dynamic>& rQ_prev_use = enough ? rQ_prev_px_f : rQ_prev_px;
            const Eigen::Matrix<double,2,Eigen::Dynamic>& rQ_curr_use = enough ? rQ_curr_px_f : rQ_curr_px;

            // recompute normalised for the used sets
            auto undist_mat = [&](const Eigen::Matrix<double,2,Eigen::Dynamic>& P){
                std::vector<cv::Point2f> in(P.cols()), out;
                for (int j=0;j<P.cols();++j) in[j]=cv::Point2f((float)P(0,j),(float)P(1,j));
                cv::undistortPoints(in, out, Kcv, Dcv);
                Eigen::Matrix<double,2,Eigen::Dynamic> N(2, (int)out.size());
                for (int j=0;j<(int)out.size();++j){ N(0,j)=out[j].x; N(1,j)=out[j].y; }
                return N;
            };
            Eigen::Matrix<double,2,Eigen::Dynamic> rQ_prev_n_use = undist_mat(rQ_prev_use);
            Eigen::Matrix<double,2,Eigen::Dynamic> rQ_curr_n_use = undist_mat(rQ_curr_use);

            // ---- Sky-only rotation prior (Kabsch on rays) ----
            bool have_sky_rot = false;
            Eigen::Vector3d dRPY_sky = Eigen::Vector3d::Zero();
            int nin_sky = 0;
            {
                std::vector<Eigen::Vector3d> v_prev, v_curr;
                v_prev.reserve(rQ_prev_n_use.cols()); v_curr.reserve(rQ_prev_n_use.cols());
                for (int j = 0; j < rQ_prev_n_use.cols(); ++j) {
                    const double s_ref  = ground_sign * sdist_n(rQ_prev_n_use(0,j), rQ_prev_n_use(1,j));
                    const double s_curr = ground_sign * sdist_n(rQ_curr_n_use(0,j), rQ_curr_n_use(1,j));
                    if (s_ref < -BAND_S && s_curr < -BAND_S) {
                        Eigen::Vector3d a(rQ_prev_n_use(0,j),  rQ_prev_n_use(1,j),  1.0);  a.normalize();
                        Eigen::Vector3d b(rQ_curr_n_use(0,j), rQ_curr_n_use(1,j), 1.0);  b.normalize();
                        v_prev.push_back(a); v_curr.push_back(b);
                    }
                }
                nin_sky = (int)v_prev.size();
                if (nin_sky >= 80) {
                    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
                    for (int t = 0; t < nin_sky; ++t) H += v_curr[t] * v_prev[t].transpose();
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    const Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
                    const Eigen::Matrix3d Rhat = U * Eigen::DiagonalMatrix<double,3>(1,1,(U*V.transpose()).determinant()) * V.transpose();
                    dRPY_sky = rot2rpy(Rhat);
                    have_sky_rot = true;
                }
            }

            // ---- Build per-feature measurement (all gaps = 1) ----
            std::vector<int> gaps(rQ_curr_use.cols(), 1);
            MeasurementOutdoorFlowBundle meas_pf(i/fps, camera,
                                                 imgk_raw,
                                                 rQ_curr_use, rQ_prev_use,
                                                 gaps);

            // Inliers and solve condition
            const auto& mask = meas_pf.inlierMask();
            const int np = (int)rQ_curr_use.cols();
            const int nin = std::count(mask.begin(), mask.end(), (unsigned char)1);
            const bool ok_tracks = (np >= 180) && (nin >= 120) && (nin >= (int)std::round(0.45*np));

            if (!ok_tracks) {
                etak = etakm1;
                std::println("[k={}] weak support: nin={} / np={}. Holding pose.", k, nin, np);
            }

            // ---- Altimeter event gating (Set 1) ----
            const double t_k = i / fps;
            const auto alt_past = altAGLPastIdx(djiVideoCaption, t_k);
            bool   alt_event     = false;
            double alt_agl_meas  = std::numeric_limits<double>::quiet_NaN();
            if (alt_past) {
                alt_agl_meas = alt_past->first;
                alt_event    = (alt_past->second != last_alt_idx);
            }

            // Degeneracy proxies
            int cnt_near=0, cnt_ground=0, cnt_all=0;
            const double NEAR_N   = 35.0 / fx;
            const double GROUND_N = 50.0 / fx;
            for (int j = 0; j < np; ++j) if (mask[j]) {
                const double s = ground_sign * sdist_n(rQ_curr_n_use(0,j), rQ_curr_n_use(1,j));
                if (std::abs(s) < NEAR_N) ++cnt_near;
                if (s > GROUND_N)         ++cnt_ground;
                ++cnt_all;
            }
            const double frac_near   = (cnt_all>0) ? (double)cnt_near   / (double)cnt_all : 0.0;
            const double ground_frac = (cnt_all>0) ? (double)cnt_ground / (double)cnt_all : 0.0;
            const bool weak_ground_support = (ground_frac < 0.35);

            // ---- Cost function (Set 1 + sky rotation prior restored) ----
            auto costFunc = [&](const Eigen::VectorXd & etak_cur, Eigen::VectorXd & g, Eigen::MatrixXd & H)
            {
                double f = meas_pf.costOdometry(etak_cur, etakm1, g, H);

                // dynamic flow weight
                double w_flow;
                if      (frac_near > 0.60) w_flow = 0.65;
                else if (frac_near > 0.40) w_flow = 0.75;
                else if (weak_ground_support) w_flow = 0.85;
                else                        w_flow = 0.95;
                f *= w_flow; g *= w_flow; H *= w_flow;

                // altimeter event residual (scalar)
                if (alt_event) {
                    constexpr double sigma_alt_m = 0.6;
                    const double M    = (double)nin;
                    const double wAlt = 0.8 * M;
                    const double invR = wAlt / (sigma_alt_m * sigma_alt_m);
                    const double r = etak_cur(2) + alt_agl_meas; // NED: D + AGL = 0
                    f     += r * r * invR;
                    g(2)  += 2.0 * r * invR;
                    H(2,2)+= 2.0 * invR;
                }

                // small attitude smoothness prior (always on)
                {
                    constexpr double sigma_rp_always_deg = 0.35;
                    const double sigma_rp = sigma_rp_always_deg * M_PI / 180.0;
                    const double invRrp   = 1.0 / (sigma_rp * sigma_rp);
                    const double rR = etak_cur(3) - etakm1(3);
                    const double rP = etak_cur(4) - etakm1(4);
                    f    += (rR*rR + rP*rP) * invRrp;
                    g(3) += 2.0 * rR * invRrp;  H(3,3) += 2.0 * invRrp;
                    g(4) += 2.0 * rP * invRrp;  H(4,4) += 2.0 * invRrp;
                }

                // soft att prior when ground weak & no alt event
                if (!alt_event && weak_ground_support) {
                    const double sigma_rp_deg =
                        (frac_near > 0.60) ? 0.15 :
                        (frac_near > 0.40) ? 0.22 :
                        0.35;
                    const double sigma_rp = sigma_rp_deg * M_PI / 180.0;
                    const double invRrp   = 1.0 / (sigma_rp * sigma_rp);
                    const double rR = etak_cur(3) - etakm1(3);
                    const double rP = etak_cur(4) - etakm1(4);
                    f    += (rR*rR + rP*rP) * invRrp;
                    g(3) += 2.0 * rR * invRrp;  H(3,3) += 2.0 * invRrp;
                    g(4) += 2.0 * rP * invRrp;  H(4,4) += 2.0 * invRrp;
                }

                // ---- SKY ROTATION PRIOR (restored from Set 1) ----
                if (have_sky_rot) {
                    auto wrap = [](double a){ while (a >  M_PI) a -= 2*M_PI; while (a < -M_PI) a += 2*M_PI; return a; };
                    const double dR = wrap( (etak_cur(3) - etakm1(3)) - dRPY_sky(0) );
                    const double dP = wrap( (etak_cur(4) - etakm1(4)) - dRPY_sky(1) );
                    const double dY = wrap( (etak_cur(5) - etakm1(5)) - dRPY_sky(2) );
                    const double Ms        = (double)nin_sky;
                    const double w_sky     = 0.60 * Ms;
                    const double s_rp_deg  = 0.30;
                    const double s_yaw_deg = 0.70;
                    const double invRr = w_sky / std::pow(s_rp_deg  * M_PI/180.0, 2);
                    const double invRp = invRr;
                    const double invRy = w_sky / std::pow(s_yaw_deg * M_PI/180.0, 2);
                    f    += dR*dR * invRr;  g(3) += 2.0*dR * invRr;  H(3,3) += 2.0*invRr;
                    f    += dP*dP * invRp;  g(4) += 2.0*dP * invRp;  H(4,4) += 2.0*invRp;
                    f    += dY*dY * invRy;  g(5) += 2.0*dY * invRy;  H(5,5) += 2.0*invRy;
                }

                return f;
            };

            // ---- Optimise ----
            etak = etakm1;
            if (alt_event && std::isfinite(alt_agl_meas)) {
                // prime D with latest AGL (NED convention D = -AGL at start)
                etak(2) = -alt_agl_meas;
            }
            const int verbosity = 1;
            int ret = funcmin::NewtonTrust(costFunc, etak, verbosity);

            if (ret == 0 && alt_event) {
                last_alt_idx = alt_past->second;
                last_alt_agl = alt_agl_meas;
            }
            if (ret != 0 || !std::isfinite(etak.sum())) {
                etak = etakm1;
            }

            // ---- Visualisation (Set 2 style) ----
            cv::Mat imgout;
            cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor);

            Eigen::VectorXd x(18); x.setZero();
            x.segment<6>(6)  = etak;
            x.segment<6>(12) = etakm1;
            Eigen::Matrix<double,2,Eigen::Dynamic> rQ_pred_px = meas_pf.predictedFeatures(x, system);

            // gate (same as Set 1)
            const double sigma_flow_px = 2.0, nSigma_gate = 3.0;
            auto gate2 = GaussianInfo<double>::fromInfo(Eigen::Vector2d::Zero(),
                                                        (1.0/(sigma_flow_px*sigma_flow_px))*Eigen::Matrix2d::Identity());
            std::vector<unsigned char> gateMask(np,0);
            for (int j=0;j<np;++j){
                Eigen::Vector2d r; r << (rQ_curr_use(0,j)-rQ_pred_px(0,j)),
                                     (rQ_curr_use(1,j)-rQ_pred_px(1,j));
                gateMask[j] = gate2.isWithinConfidenceRegion(r, nSigma_gate) ? 1 : 0;
            }

            // draw arrows
            for (int j=0;j<np;++j){
                cv::Point2d p0(rQ_prev_use(0,j)/divisor, rQ_prev_use(1,j)/divisor);
                cv::Point2d pm(rQ_curr_use(0,j)/divisor, rQ_curr_use(1,j)/divisor);
                cv::Point2d ph(rQ_pred_px  (0,j)/divisor, rQ_pred_px  (1,j)/divisor);

                // predicted (blue)
                cv::arrowedLine(imgout, p0, ph, cv::Scalar(255,0,0), 2, cv::LINE_AA, 0, 0.25);

                // measured (green/red)
                const bool ok = (meas_pf.inlierMask()[j] != 0) && (gateMask[j] != 0);
                cv::arrowedLine(imgout, p0, pm, ok ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2, cv::LINE_AA, 0, 0.25);
            }

            plotGroundPlane(imgout, etak, camera, divisor);
            plotHorizon(imgout, etak, camera, divisor);
            plotCompass(imgout, etak, camera, divisor);
            plotEpipole(imgout, etak, etakm1, camera, divisor);

            // telemetry (measured AGL shows last accepted sample only)
            const std::optional<double> alt_meas = std::isfinite(last_alt_agl) ? std::optional<double>(last_alt_agl) : std::nullopt;
            plotTelemetry(imgout, etak, alt_meas, divisor);

            // ALT EVENT label at downscaled coords
            if (alt_event) {
                cv::putText(imgout, "ALT EVENT", {612,13}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2, cv::LINE_AA);
            } else {
                cv::putText(imgout, "ALT EVENT", {612,13}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2, cv::LINE_AA);
            }

            cv::imshow("Visual odometry demo", imgout);
            char key = cv::waitKey(1);
            if (key == 'q') { std::println("Key '{}' pressed. Exiting.", key); break; }

            if (doExport) bufferedVideoWriter.write(imgout);

            // seed next
            rQ_prev_px.resize(2, rQ_curr_use.cols());
            rQ_prev_px = rQ_curr_use;
        }

        imgk_raw.copyTo(imgkm1_raw);
        etakm1 = etak;
        ++k;
    }

    if (doExport) bufferedVideoWriter.stop();
    bufferedVideoReader.stop();
}

Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
{
    const double h  = caption0.altitude;
    const double ga = h - 7.0;
    Eigen::Vector6d eta0;
    eta0 << 0.0, 0.0, -ga, 0.0, 0.0, 0.0; // NED, D = -AGL
    return eta0;
}

// ---- Set 2 plotting style ----

void plotGroundPlane(cv::Mat & img,
                     const Eigen::Vector6d & etak,
                     const Camera & camera,
                     const int & divisor)
{
    const double Dplane        = 0.0;
    const double span          = 1000.0;
    const double spacing       = 100.0;
    const double segmentLen    = 10.0;
    const cv::Scalar gridCol   = cv::Scalar(0,0,0);
    const double thickness     = 1.0;
    const double thicknessAxis = 1.0;
    const int    lineType      = cv::LINE_AA;
    const double N0            = 0.0;
    const double E0            = 0.0;

    const int th  = std::max<int>(1, std::lround(thickness));
    const int tha = std::max<int>(1, std::lround(thicknessAxis));

    const cv::Rect roi(-1, -1, img.cols + 2, img.rows + 2);

    const Eigen::Vector3d rBNn = etak.head<3>();
    const Eigen::Matrix3d Rnb  = rpy2rot(etak.tail<3>());
    Pose<double> Tnb(Rnb, rBNn);

    const Pose<double> Tnc      = Tnb * camera.Tbc;
    const Eigen::Matrix3d R_WC  = Tnc.rotationMatrix;
    const Eigen::Vector3d r_WC  = Tnc.translationVector;

    auto camDepthZ = [&](double N, double E)->double {
        const Eigen::Vector3d PW(N, E, Dplane);
        const Eigen::Vector3d PC = R_WC.transpose() * (PW - r_WC);
        return PC.z();
    };
    auto inFOV = [&](double N, double E)->bool {
        return camera.isWorldWithinFOV(cv::Vec3d(N, E, Dplane), Tnb);
    };
    auto projectFront = [&](double N, double E, cv::Point& p)->bool {
        if (camDepthZ(N, E) <= 0.0) return false;
        cv::Vec2d uv = camera.worldToPixel(cv::Vec3d(N, E, Dplane), Tnb);
        if (!std::isfinite(uv[0]) || !std::isfinite(uv[1])) return false;
        p.x = cvRound(uv[0] / divisor);
        p.y = cvRound(uv[1] / divisor);
        return true;
    };

    auto drawFamily = [&](bool Nconst, double fixedVal, int thicknessPixels)
    {
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

        for (double s0 = sBegin; s0 < sEnd; s0 += segmentLen)
        {
            const double s1 = std::min(s0 + segmentLen, sEnd);
            const bool f0 = fovCheck(s0);
            const bool f1 = fovCheck(s1);
            if (!(f0 || f1)) continue;

            cv::Point p0, p1;
            const bool ok0 = project(s0, p0);
            const bool ok1 = project(s1, p1);

            if (ok0 && ok1)
            {
                cv::Point a = p0, b = p1;
                if (cv::clipLine(roi, a, b))
                    cv::line(img, a, b, gridCol, thicknessPixels, lineType);
            }
            else
            {
                const double sm = 0.5 * (s0 + s1);
                cv::Point pm;
                if (fovCheck(sm) && project(sm, pm))
                {
                    if (ok0) { cv::Point a=p0, b=pm; if (cv::clipLine(roi, a, b)) cv::line(img, a, b, gridCol, thicknessPixels, lineType); }
                    if (ok1) { cv::Point a=pm, b=p1; if (cv::clipLine(roi, a, b)) cv::line(img, a, b, gridCol, thicknessPixels, lineType); }
                }
            }
        }
    };

    const double NStart = std::floor((N0 - span)/spacing) * spacing;
    const double NEnd   = std::floor((N0 + span)/spacing) * spacing;
    for (double N = NStart; N <= NEnd + 1e-9; N += spacing)
    {
        const int t = (std::abs(N) < 1e-9) ? tha : th;
        drawFamily(/*Nconst=*/true, N, t);
    }

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
    Eigen::Matrix3d K; cv::cv2eigen(camera.cameraMatrix, K);
    const Eigen::Matrix3d KinvT = K.inverse().transpose();
    const Eigen::Matrix3d Rcw = Rcw_from(etak, camera);
    const Eigen::Vector3d nw(0,0,1); // simple Set 2 form
    const Eigen::Vector3d l = KinvT * (Rcw * nw);

    const double a = l.x(), b = l.y(), c = l.z();
    const int Wfull = img.cols * divisor;
    const int Hfull = img.rows * divisor;

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
    const Eigen::Matrix3d Rcw = Rcw_from(etak, camera);

    Eigen::Matrix3d K; cv::cv2eigen(camera.cameraMatrix, K);

    const int W = img.cols, H = img.rows;
    auto inside = [&](double x, double y){ return (x>=0 && x<W && y>=0 && y<H); };

    auto projDeg = [&](double deg, Eigen::Vector3d& d_c, cv::Point2d& p)->bool {
        const double a = deg * M_PI / 180.0;
        Eigen::Vector3d d_w(std::cos(a), std::sin(a), 0.0);
        d_c = Rcw * d_w;
        if (!std::isfinite(d_c.z()) || d_c.z() <= 1e-9) return false;
        Eigen::Vector2d pix = camera.vectorToPixel(d_c);
        p.x = pix.x()/divisor; p.y = pix.y()/divisor;
        return std::isfinite(p.x) && std::isfinite(p.y) && inside(p.x, p.y);
    };

    auto tangentAtDeg = [&](double deg)->cv::Point2d {
        constexpr double dth = 2.0;
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

    const int fontFace       = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale   = 1.0;
    const int textThickness  = 1;
    const int n_off_px       = -50;

    for (const auto& P : pairs) {
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
        } else { continue; }

        const cv::Point2d t = tangentAtDeg(useDeg);
        const cv::Point2d n(-t.y, t.x);

        cv::Point p_up(cvRound(p.x + n.x*(n_off_px-2)), cvRound(p.y + n.y*(n_off_px-2)));
        cv::line(img, p, p_up, cv::Scalar(0,0,0), 1, cv::LINE_AA);

        int base=0; cv::Size sz = cv::getTextSize(name, fontFace, fontScale, textThickness, &base);
        cv::Point pos(cvRound(p.x - sz.width*0.5 + n_off_px*n.x),
                      cvRound(p.y + sz.height*0.5 + n_off_px*n.y));
        drawBlackText(img, name, pos, fontFace, fontScale, textThickness);
    }
}

static void plotEpipole(cv::Mat & img,
                        const Eigen::Vector6d & etak,
                        const Eigen::Vector6d & etakm1,
                        const Camera & camera,
                        const int & divisor)
{
    Eigen::Matrix3d K; cv::cv2eigen(camera.cameraMatrix, K);
    const Eigen::Matrix3d Rcw_k = Rcw_from(etak, camera);

    const Eigen::Vector3d Ck   = etak.head<3>();
    const Eigen::Vector3d Ckm1 = etakm1.head<3>();

    const Eigen::Vector3d t_k = Rcw_k * (Ckm1 - Ck);
    const Eigen::Vector3d e = K * t_k;
    if (std::isfinite(e.z()) && std::abs(e.z()) > 1e-9) {
        const cv::Point2d pe((e.x()/e.z())/divisor, (e.y()/e.z())/divisor);
        const cv::Scalar ORANGE(0,165,255);
        cv::circle(img, pe, 14, ORANGE, 3, cv::LINE_AA);
    }

    // velocity direction marker
    const Eigen::Vector3d vc = Rcw_k * (Ck - Ckm1);
    if (std::isfinite(vc.squaredNorm()) && vc.squaredNorm() > 1e-16) {
        Eigen::Vector3d d = vc; if (d.z() <= 0) d = -d;
        const Eigen::Vector2d pix = camera.vectorToPixel(d);
        cv::Point2d pv(pix.x()/divisor, pix.y()/divisor);
        const cv::Scalar ORANGE(0,165,255);
        cv::circle(img, pv, 8, ORANGE, 2, cv::LINE_AA);
        cv::putText(img, "v", {cvRound(pv.x)+6, cvRound(pv.y)-6},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1, cv::LINE_AA);
    }
}

static void plotTelemetry(cv::Mat& img,
                          const Eigen::Vector6d& etak,
                          const std::optional<double>& alt_agl,
                          const int& /*divisor*/)
{
    auto putTextOutlined = [](cv::Mat& im, const std::string& txt, cv::Point org, const cv::Scalar& col)
    {
        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const double scale = 0.6;
        const int thick = 2;
        cv::putText(im, txt, org, font, scale, cv::Scalar(0,0,0), thick+2, cv::LINE_AA);
        cv::putText(im, txt, org, font, scale, col, thick, cv::LINE_AA);
    };

    const double est_agl = -etak(2);
    const double est_N   =  etak(0);
    const double est_E   =  etak(1);

    int x0 = 12;
    int y  = 26;
    const int lh = 24;

    const cv::Scalar BLUE (255, 0, 0);
    const cv::Scalar GREEN(0, 255, 0);

    putTextOutlined(img, cv::format("Est Alt (AGL): %.1f m", est_agl), {x0, y}, BLUE); y += lh;
    if (alt_agl) {
        putTextOutlined(img, cv::format("Meas Alt (AGL): %.1f m", *alt_agl), {x0, y}, GREEN); y += lh;
    }
    putTextOutlined(img, cv::format("Est NE: N=%.1f m, E=%.1f m", est_N, est_E), {x0, y}, BLUE);
}
