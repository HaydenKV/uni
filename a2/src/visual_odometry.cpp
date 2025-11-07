// visual_odometry.cpp — Set 1 behaviour + Set 2 plotting/flow/tuning (adaptive n restored)

#include <print>
#include <cmath>
#include <optional>
#include <algorithm>
#include <array>
#include <deque>
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

// ── FWD DECLS ─────────────────────────────────────────────────────────────────
static Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0);
static void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor);
static void plotTelemetry(cv::Mat& img, const Eigen::Vector6d& etak, const std::optional<double>& alt_agl, const int& divisor);

// ── Small helpers ────────────────────────────────────────────────────────────
static inline Eigen::Matrix3d Rcw_from(const Eigen::Vector6d& eta, const Camera& cam) {
    const Eigen::Matrix3d Rnb = rpy2rot(eta.tail<3>());
    const Eigen::Matrix3d Rbc = cam.Tbc.rotationMatrix;
    return (Rnb * Rbc).transpose(); // world→camera
}

// Latest caption sample at or before t
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

// ── Ring buffer for grayscale frames (for adaptive backtracking) ─────────────
struct FrameBuf {
    double t = 0.0;
    cv::Mat gray;
    bool valid = false;
};
static constexpr int K_ADAPT = 6;  // consider up to 6 frames back
static std::deque<FrameBuf> g_hist; // newest past at front: g_hist[0] == k-1

// Tunables for “good” optical flow length (px, full-res)
static double kFlowMin = 2.0;     // below this => uninformative
static double kFlowMax = 25.0;    // above this => unreliable

// ── Main ─────────────────────────────────────────────────────────────────────
void runVisualOdometryFromVideo(const std::filesystem::path & videoPath,
                                const std::filesystem::path & cameraPath,
                                const std::filesystem::path & outputDirectory)
{
    // Set 2 sampling & plot scale (set to 1 to process every frame)
    int imgModulus  = 10;
    int divisor     = 2;

    assert(!videoPath.empty());

    // Subtitles
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

    // Set 2 style small gimbal biases
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
    int nFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::println("Input video: {}", videoPath.string());
    std::println("Subtitle file: {}", subtitlePath.string());
    std::println("Total number of frames: {}", nFrames);
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
        double outputFps = fps / imgModulus;
        int codec = cv::VideoWriter::fourcc('m','p','4','v');
        if (videoOut.open(outputPath.string(), codec, outputFps, frameSize)) {
            bufferedVideoWriter.start(videoOut);
            std::println("Export visualisation to {}", outputPath.string());
        } else {
            std::println("Warning: could not open video writer '{}'. Export disabled.", outputPath.string());
            doExport = false;
        }
    }

    // System (dummy)
    auto p0 = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), Eigen::MatrixXd::Zero(18, 18));
    SystemVisualNav system(p0);

    // State
    Eigen::VectorXd etakm1(6), etak(6);
    etak = getInitialPose(djiVideoCaption[0]);

    // Frame mats
    cv::Mat imgk_raw, imgkm1_raw;

    // Two-frame seed (k−1)
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQ_prev_px;

    // Alt event tracking
    static int    last_alt_idx  = -1;
    static double last_alt_agl  = std::numeric_limits<double>::quiet_NaN();

    for (int i = 0, k = 0;; ++i)
    {
        imgk_raw = bufferedVideoReader.read();
        if (imgk_raw.empty()) break;

        if (i % imgModulus != 0) continue; // sampling like Set 2

        // Current grayscale
        cv::Mat gray_k;
        if (imgk_raw.channels() == 3) cv::cvtColor(imgk_raw, gray_k, cv::COLOR_BGR2GRAY);
        else gray_k = imgk_raw;

        if (k == 0)
        {
            // Bootstrap ring buffer
            g_hist.push_front(FrameBuf{ i / fps, gray_k.clone(), true });
            while ((int)g_hist.size() > K_ADAPT) g_hist.pop_back();

            imgk_raw.copyTo(imgkm1_raw);
            etakm1 = etak;
            ++k;
            continue;
        }

        // ── (1) Generate k−1→k tracks with Set 2’s path ─────────────────────
        MeasurementOutdoorFlowBundle meas2(i/fps, camera, imgk_raw, imgkm1_raw, rQ_prev_px);
        const Eigen::Matrix<double,2,Eigen::Dynamic>& rQ_curr_px_all = meas2.trackedCurrentFeatures();
        rQ_prev_px = meas2.trackedPreviousFeatures();

        // Build vector of cv::Point2f for current points
        int np_full = (int)rQ_curr_px_all.cols();
        const int np_orig = np_full;
        std::vector<cv::Point2f> ptsCurr(np_full);
        for (int j = 0; j < np_full; ++j) {
            ptsCurr[j].x = (float)rQ_curr_px_all(0,j);
            ptsCurr[j].y = (float)rQ_curr_px_all(1,j);
        }

        // ── (1.1) (Optional) Sky quota top-up like Set 1 (kept compact) ─────
        {
            cv::Mat Kcv = camera.cameraMatrix, Dcv = camera.distCoeffs;

            // Horizon in normalized plane from etakm1
            const Eigen::Matrix3d Rcw_n = Rcw_from(etakm1, camera);
            const Eigen::Vector3d n_w(0,0,-1);
            const Eigen::Vector3d l_n = Rcw_n * n_w;
            auto sdist_n = [&](double xn, double yn){
                const double a=l_n.x(), b=l_n.y(), c=l_n.z();
                const double den = std::sqrt(a*a + b*b) + 1e-12;
                return (a*xn + b*yn + c) / den;
            };

            // sign via bottom-centre
            const double cx_px = Kcv.at<double>(0,2);
            const double Hfull = (double)imgk_raw.rows;
            std::vector<cv::Point2f> bc_in(1), bc_out;
            bc_in[0] = cv::Point2f((float)cx_px, (float)(Hfull-1.0));
            cv::undistortPoints(bc_in, bc_out, Kcv, Dcv);
            const double ground_sign = (sdist_n(bc_out[0].x, bc_out[0].y) >= 0.0) ? 1.0 : -1.0;

            // count current SKY pts (normalized)
            std::vector<cv::Point2f> ptsCurr_n;
            {
                std::vector<cv::Point2f> tmp;
                cv::undistortPoints(ptsCurr, tmp, Kcv, Dcv);
                ptsCurr_n = std::move(tmp);
            }
            const double fx = Kcv.at<double>(0,0);
            const double BAND_S = 40.0 / fx;
            int sky_count = 0;
            for (int j=0;j<np_full;++j) {
                const double s = ground_sign * sdist_n(ptsCurr_n[j].x, ptsCurr_n[j].y);
                if (s < -BAND_S) ++sky_count;
            }

            constexpr int SKY_MIN=500, SKY_MAX=600, SKY_TARGET=575;
            if (sky_count < SKY_MIN && !g_hist.empty()) {
                // pixel-space horizon from l_n
                const double fy = Kcv.at<double>(1,1);
                const double cx = Kcv.at<double>(0,2), cy = Kcv.at<double>(1,2);
                const double a_px = l_n.x()/fx;
                const double b_px = l_n.y()/fy;
                const double c_px = l_n.z() - l_n.x()*cx/fx - l_n.y()*cy/fy;

                const int W = imgk_raw.cols, H = imgk_raw.rows;
                cv::Mat skyMask(H, W, CV_8UC1, cv::Scalar(0));
                const int margin = 35;
                for (int x=0;x<W;++x){
                    double y_h = (std::abs(b_px) > 1e-9) ? -(a_px * x + c_px) / b_px : -1e9;
                    int ymax = (int)std::floor(y_h - margin);
                    ymax = std::min(std::max(ymax, 0), H);
                    if (ymax > 0) skyMask.rowRange(0, ymax).col(x).setTo(255);
                }

                const int want = std::min(SKY_TARGET - sky_count, SKY_MAX - sky_count);
                if (want > 0) {
                    std::vector<cv::Point2f> sky_pts;
                    cv::goodFeaturesToTrack(gray_k, sky_pts, want, 0.01, 6.0, skyMask);
                    auto tooClose = [&](const cv::Point2f& p){
                        for (const auto& q : ptsCurr) {
                            const float dx=p.x-q.x, dy=p.y-q.y;
                            if (dx*dx+dy*dy < 36.0f) return true;
                        }
                        return false;
                    };
                    std::vector<cv::Point2f> sky_keep; sky_keep.reserve(sky_pts.size());
                    for (const auto& p: sky_pts) if (!tooClose(p)) sky_keep.push_back(p);
                    ptsCurr.insert(ptsCurr.end(), sky_keep.begin(), sky_keep.end());
                    np_full = (int)ptsCurr.size();
                }
            }
        }

        // ── (2) Adaptive backtracking into k−n, n ∈ [1..K_ADAPT] ────────────
        const cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.01);
        const cv::Size winSize(21, 21);
        const float fb_thresh = 2.0f;
        const int Ncand = std::min((int)g_hist.size(), K_ADAPT);

        std::vector<std::vector<cv::Point2f>> backPtsPerN(std::max(0,Ncand));
        std::vector<std::vector<unsigned char>> okPerN(std::max(0,Ncand));

        for (int n = 1; n <= Ncand; ++n)
        {
            const cv::Mat& gray_old = g_hist[n-1].gray; // k-n
            backPtsPerN[n-1].assign(np_full, cv::Point2f());
            okPerN[n-1].assign(np_full, 0);

            // backward k→k−n
            std::vector<unsigned char> st_b;  std::vector<float> err_b;
            cv::calcOpticalFlowPyrLK(gray_k, gray_old, ptsCurr, backPtsPerN[n-1],
                                     st_b, err_b, winSize, 3, termcrit, 0, 1e-4f);
            // forward (FB check)
            std::vector<cv::Point2f> fwdPts(np_full);
            std::vector<unsigned char> st_f;  std::vector<float> err_f;
            cv::calcOpticalFlowPyrLK(gray_old, gray_k, backPtsPerN[n-1], fwdPts,
                                     st_f, err_f, winSize, 3, termcrit, 0, 1e-4f);

            const float fb2 = fb_thresh * fb_thresh;
            for (int j=0;j<np_full;++j) {
                bool ok = (st_b.size()>(size_t)j && st_f.size()>(size_t)j && st_b[j] && st_f[j]);
                if (ok) {
                    const cv::Point2f d = fwdPts[j] - ptsCurr[j];
                    ok = (d.dot(d) <= fb2);
                }
                okPerN[n-1][j] = ok ? 1 : 0;
            }
        }

        // Choose per-feature gap and reference
        std::vector<int>  gap_for_feat(np_full, 1);
        std::vector<cv::Point2f> refPts(np_full);
        std::vector<char> valid(np_full, 1);

        for (int j=0;j<np_full;++j)
        {
            bool assigned=false;
            for (int n=1;n<=Ncand;++n){
                if (!okPerN[n-1][j]) continue;
                const cv::Point2f& pref = backPtsPerN[n-1][j];
                const float dx = ptsCurr[j].x - pref.x;
                const float dy = ptsCurr[j].y - pref.y;
                const float flow = std::sqrt(dx*dx + dy*dy);
                if (flow >= (float)kFlowMin && flow <= (float)kFlowMax) {
                    gap_for_feat[j] = n;
                    refPts[j]       = pref;
                    assigned        = true;
                    break;
                }
            }
            if (!assigned) {
                if (Ncand>=1 && okPerN[0][j]) {
                    gap_for_feat[j] = 1;
                    refPts[j]       = backPtsPerN[0][j];
                } else if (j < np_orig) {
                    gap_for_feat[j] = 1;
                    refPts[j]       = cv::Point2f((float)rQ_prev_px(0,j),(float)rQ_prev_px(1,j));
                } else {
                    valid[j] = 0; // newly added sky point with no backtrack
                }
            }
        }

        // Assemble Eigen matrices, compress to valid
        Eigen::Matrix<double,2,Eigen::Dynamic> rQ_ref_tmp(2, np_full), rQ_curr_tmp(2, np_full);
        for (int j=0;j<np_full;++j){
            rQ_curr_tmp(0,j)=ptsCurr[j].x; rQ_curr_tmp(1,j)=ptsCurr[j].y;
            rQ_ref_tmp (0,j)=refPts[j].x;  rQ_ref_tmp (1,j)=refPts[j].y;
        }
        int nvalid=0; for (int j=0;j<np_full;++j) if (valid[j]) ++nvalid;

        Eigen::Matrix<double,2,Eigen::Dynamic> rQ_ref (2,nvalid);
        Eigen::Matrix<double,2,Eigen::Dynamic> rQ_curr(2,nvalid);
        std::vector<int> gap_keep; gap_keep.reserve(nvalid);
        { int w=0; for (int j=0;j<np_full;++j) if (valid[j]) { rQ_curr.col(w)=rQ_curr_tmp.col(j); rQ_ref.col(w)=rQ_ref_tmp.col(j); gap_keep.push_back(gap_for_feat[j]); ++w; } }
        gap_for_feat.swap(gap_keep);

        // ── (2.5) Horizon band filter (ground or sky only) ───────────────────
        cv::Mat Kcv = camera.cameraMatrix, Dcv = camera.distCoeffs;
        auto undist_mat = [&](const Eigen::Matrix<double,2,Eigen::Dynamic>& P){
            std::vector<cv::Point2f> in(P.cols()), out;
            for (int j=0;j<P.cols();++j) in[j]=cv::Point2f((float)P(0,j),(float)P(1,j));
            cv::undistortPoints(in, out, Kcv, Dcv);
            Eigen::Matrix<double,2,Eigen::Dynamic> N(2, (int)out.size());
            for (int j=0;j<(int)out.size();++j){ N(0,j)=out[j].x; N(1,j)=out[j].y; }
            return N;
        };
        Eigen::Matrix<double,2,Eigen::Dynamic> rQ_ref_n  = undist_mat(rQ_ref);
        Eigen::Matrix<double,2,Eigen::Dynamic> rQ_curr_n = undist_mat(rQ_curr);

        const Eigen::Matrix3d Rcw_n = Rcw_from(etakm1, camera);
        const Eigen::Vector3d n_w(0,0,-1);
        const Eigen::Vector3d l_n = Rcw_n * n_w;
        auto sdist_n = [&](double xn, double yn){
            const double a=l_n.x(), b=l_n.y(), c=l_n.z();
            const double den = std::sqrt(a*a + b*b) + 1e-12;
            return (a*xn + b*yn + c) / den;
        };
        // ground sign via bottom-centre
        {
            const double cx_px = Kcv.at<double>(0,2);
            const double Hfull = (double)imgk_raw.rows;
            std::vector<cv::Point2f> bc_in(1), bc_out;
            bc_in[0] = cv::Point2f((float)cx_px, (float)(Hfull-1.0));
            cv::undistortPoints(bc_in, bc_out, Kcv, Dcv);
            const double sign = (sdist_n(bc_out[0].x, bc_out[0].y) >= 0.0) ? 1.0 : -1.0;
            // thresholds in normalised units (~40 px / fx)
            const double fx = Kcv.at<double>(0,0);
            const double BAND_G = 40.0 / fx;
            const double BAND_S = 40.0 / fx;

            std::vector<int> keep; keep.reserve((int)rQ_curr.cols());
            for (int j=0;j<rQ_curr.cols();++j){
                const double s_ref  = sign * sdist_n(rQ_ref_n (0,j), rQ_ref_n (1,j));
                const double s_curr = sign * sdist_n(rQ_curr_n(0,j), rQ_curr_n(1,j));
                const bool is_ground = (s_ref > +BAND_G) && (s_curr > +BAND_G);
                const bool is_sky    = (s_ref < -BAND_S) && (s_curr < -BAND_S);
                if (is_ground || is_sky) keep.push_back(j);
            }

            const int np_now = (int)rQ_curr.cols();
            const int MIN_FEAT=90; const double MIN_FRAC=0.22;
            auto apply_keep = [&](const std::vector<int>& K){
                Eigen::Matrix<double,2,Eigen::Dynamic> a(2,K.size()), b(2,K.size());
                std::vector<int> g; g.reserve(K.size());
                for (size_t kx=0;kx<K.size();++kx){ const int j=K[kx]; a.col(kx)=rQ_ref.col(j); b.col(kx)=rQ_curr.col(j); g.push_back(gap_for_feat[j]); }
                rQ_ref.swap(a); rQ_curr.swap(b); gap_for_feat.swap(g);
                rQ_ref_n  = undist_mat(rQ_ref);
                rQ_curr_n = undist_mat(rQ_curr);
            };
            if ((int)keep.size() >= std::max(MIN_FEAT, (int)std::round(MIN_FRAC*np_now))) {
                apply_keep(keep);
            } else if ((int)keep.size() >= 60) {
                apply_keep(keep);
            }
        }

        // ── Sky-only rotation prior (Kabsch on rays) ─────────────────────────
        bool have_sky_rot=false; Eigen::Vector3d dRPY_sky=Eigen::Vector3d::Zero(); int nin_sky=0;
        {
            const double fx = Kcv.at<double>(0,0);
            const double BAND_S = 40.0 / fx;
            // reuse sign & sdist_n defined above
            // compute sign again (cheap)
            const double cx_px = Kcv.at<double>(0,2);
            const double Hfull = (double)imgk_raw.rows;
            std::vector<cv::Point2f> bc_in(1), bc_out;
            bc_in[0] = cv::Point2f((float)cx_px, (float)(Hfull-1.0));
            cv::undistortPoints(bc_in, bc_out, Kcv, Dcv);
            const double sign = (sdist_n(bc_out[0].x, bc_out[0].y) >= 0.0) ? 1.0 : -1.0;

            std::vector<Eigen::Vector3d> v_prev, v_curr;
            const int npf=(int)rQ_ref_n.cols();
            v_prev.reserve(npf); v_curr.reserve(npf);
            for (int j=0;j<npf;++j){
                const double s_ref  = sign * sdist_n(rQ_ref_n (0,j), rQ_ref_n (1,j));
                const double s_curr = sign * sdist_n(rQ_curr_n(0,j), rQ_curr_n(1,j));
                if (s_ref < -BAND_S && s_curr < -BAND_S) {
                    Eigen::Vector3d a(rQ_ref_n(0,j),  rQ_ref_n(1,j),  1.0); a.normalize();
                    Eigen::Vector3d b(rQ_curr_n(0,j), rQ_curr_n(1,j), 1.0); b.normalize();
                    v_prev.push_back(a); v_curr.push_back(b);
                }
            }
            nin_sky = (int)v_prev.size();
            if (nin_sky >= 80) {
                Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
                for (int t=0;t<nin_sky;++t) H += v_curr[t] * v_prev[t].transpose();
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU|Eigen::ComputeFullV);
                const Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
                const Eigen::Matrix3d Rhat = U * Eigen::DiagonalMatrix<double,3>(1,1,(U*V.transpose()).determinant()) * V.transpose();
                dRPY_sky = rot2rpy(Rhat);
                have_sky_rot = true;
            }
        }

        // ── (3) Build per-feature measurement with gap_for_feat ──────────────
        const double t_k = i / fps;
        MeasurementOutdoorFlowBundle meas_pf(t_k, camera, imgk_raw, rQ_curr, rQ_ref, gap_for_feat);

        // Inliers and solve check
        const auto& mask = meas_pf.inlierMask();
        const int np = (int)rQ_curr.cols();
        const int nin = std::count(mask.begin(), mask.end(), (unsigned char)1);
        const bool ok_tracks = (np >= 180) && (nin >= 120) && (nin >= (int)std::round(0.45*np));
        if (!ok_tracks) {
            etak = etakm1;
            std::println("[k={}] weak support: nin={} / np={}. Holding pose.", k, nin, np);
        }

        // Altimeter event gating (new sample only)
        const auto alt_past = altAGLPastIdx(djiVideoCaption, t_k);
        bool   alt_event     = false;
        double alt_agl_meas  = std::numeric_limits<double>::quiet_NaN();
        if (alt_past) {
            alt_agl_meas = alt_past->first;
            alt_event    = (alt_past->second != last_alt_idx);
        }

        // Degeneracy metrics (normalized plane)
        int cnt_near=0, cnt_ground=0, cnt_all=0;
        {
            cv::Mat Kcv = camera.cameraMatrix;
            const double fx = Kcv.at<double>(0,0);
            const double NEAR_N = 35.0 / fx;
            const double GROUND_N = 50.0 / fx;

            // recompute normalized versions for used sets (already have)
            // need sign & sdist_n; reuse from above by recomputing sign
            const double cx_px = Kcv.at<double>(0,2);
            const double Hfull = (double)imgk_raw.rows;
            std::vector<cv::Point2f> bc_in(1), bc_out;
            bc_in[0] = cv::Point2f((float)cx_px, (float)(Hfull-1.0));
            cv::undistortPoints(bc_in, bc_out, Kcv, camera.distCoeffs);
            const double sign = (sdist_n(bc_out[0].x, bc_out[0].y) >= 0.0) ? 1.0 : -1.0;

            for (int j=0;j<np;++j) if (mask[j]) {
                const double s = sign * sdist_n(rQ_curr_n(0,j), rQ_curr_n(1,j));
                if (std::abs(s) < NEAR_N) ++cnt_near;
                if (s > GROUND_N)         ++cnt_ground;
                ++cnt_all;
            }
        }
        const double frac_near   = (cnt_all>0) ? (double)cnt_near   / (double)cnt_all : 0.0;
        const double ground_frac = (cnt_all>0) ? (double)cnt_ground / (double)cnt_all : 0.0;
        const bool weak_ground_support = (ground_frac < 0.35);

        // Cost function (Set 1)
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

            // altimeter (new sample only), weight vs inliers
            if (alt_event) {
                constexpr double sigma_alt_m = 0.6;
                const double M    = (double)nin;
                const double wAlt = 0.8 * M;
                const double invR = wAlt / (sigma_alt_m * sigma_alt_m);
                const double r = etak_cur(2) + alt_agl_meas; // NED D + AGL = 0
                f     += r * r * invR;
                g(2)  += 2.0 * r * invR;
                H(2,2)+= 2.0 * invR;
            }

            // small always-on roll/pitch smoothness
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

            // extra soft att prior if ground weak and no alt event
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

            // Sky rotation prior
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

        // Minimise
        etak = etakm1;
        if (alt_event && std::isfinite(alt_agl_meas)) {
            // Same as Set 1: prime D with latest AGL (won’t “stick” now that D is partially observable again)
            etak(2) = -alt_agl_meas;
        }
        const int verbosity = 1;
        int ret = funcmin::NewtonTrust(costFunc, etak, verbosity);

        if (ret == 0 && alt_event) {
            last_alt_idx = alt_past->second;
            last_alt_agl = alt_agl_meas;
        }
        if (ret != 0 || !std::isfinite(etak.sum())) {
            etak = etakm1; // reject bad update
        }

        // ── Visualisation (Set 2 style) ──────────────────────────────────────
        cv::Mat imgout;
        cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor);

        Eigen::VectorXd x(18); x.setZero();
        x.segment<6>(6)  = etak;
        x.segment<6>(12) = etakm1;
        Eigen::Matrix<double,2,Eigen::Dynamic> rQ_pred_px = meas_pf.predictedFeatures(x, system);

        // Confidence gate (same as Set 1)
        const double sigma_flow_px = 2.0, nSigma_gate = 3.0;
        auto gate2 = GaussianInfo<double>::fromInfo(Eigen::Vector2d::Zero(),
                                                    (1.0/(sigma_flow_px*sigma_flow_px))*Eigen::Matrix2d::Identity());
        std::vector<unsigned char> gateMask(np,0);
        for (int j=0;j<np;++j){
            Eigen::Vector2d r; r << (rQ_curr(0,j)-rQ_pred_px(0,j)),
                                 (rQ_curr(1,j)-rQ_pred_px(1,j));
            gateMask[j] = gate2.isWithinConfidenceRegion(r, nSigma_gate) ? 1 : 0;
        }

        // Draw arrows (pred=blue, ok=green, else red)
        for (int j=0;j<np;++j){
            cv::Point2d p0(rQ_ref(0,j)/divisor,  rQ_ref(1,j)/divisor);
            cv::Point2d pm(rQ_curr(0,j)/divisor, rQ_curr(1,j)/divisor);
            cv::Point2d ph(rQ_pred_px(0,j)/divisor, rQ_pred_px(1,j)/divisor);

            cv::arrowedLine(imgout, p0, ph, cv::Scalar(255,0,0), 2, cv::LINE_AA, 0, 0.25);
            const bool ok = (meas_pf.inlierMask()[j] != 0) && (gateMask[j] != 0);
            cv::arrowedLine(imgout, p0, pm, ok ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2, cv::LINE_AA, 0, 0.25);
        }

        plotGroundPlane(imgout, etak, camera, divisor);
        plotHorizon(imgout, etak, camera, divisor);
        plotCompass(imgout, etak, camera, divisor);
        plotEpipole(imgout, etak, etakm1, camera, divisor);

        // Telemetry: show last accepted measured AGL only
        const std::optional<double> alt_meas = std::isfinite(last_alt_agl) ? std::optional<double>(last_alt_agl) : std::nullopt;
        plotTelemetry(imgout, etak, alt_meas, divisor);

        // ALT EVENT banner (downscaled coords)
        if (alt_event) {
            cv::putText(imgout, "ALT EVENT", {612,13}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2, cv::LINE_AA);
        } else {
            cv::putText(imgout, "ALT EVENT", {612,13}, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2, cv::LINE_AA);
        }

        cv::imshow("Visual odometry demo", imgout);
        char key = cv::waitKey(1);
        if (key == 'q') { std::println("Key '{}' pressed. Exiting.", key); break; }

        if (doExport) bufferedVideoWriter.write(imgout);

        // Seed next two-frame tracker with current set at k (gap=1 seeds)
        rQ_prev_px.resize(2, rQ_curr.cols());
        rQ_prev_px = rQ_curr;

        // Update ring buffer; roll forward
        g_hist.push_front(FrameBuf{ i / fps, gray_k.clone(), true });
        while ((int)g_hist.size() > K_ADAPT) g_hist.pop_back();

        imgk_raw.copyTo(imgkm1_raw);
        etakm1 = etak;
        ++k;
    }

    if (doExport) bufferedVideoWriter.stop();
    bufferedVideoReader.stop();
}

// ── Initial pose from caption 0 (unchanged) ──────────────────────────────────
Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
{
    const double h  = caption0.altitude;
    const double ga = h - 7.0; // AGL
    Eigen::Vector6d eta0;
    eta0 << 0.0, 0.0, -ga, 0.0, 0.0, 0.0; // NED: D = -AGL
    return eta0;
}

// ── Set 2 plotting style (unchanged from previous merge) ─────────────────────
void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
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
                cv::Point a=p0, b=p1;
                if (cv::clipLine(roi, a, b))
                    cv::line(img, a, b, gridCol, thicknessPixels, lineType);
            }
            else
            {
                const double sm = 0.5*(s0+s1);
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
    const Eigen::Vector3d nw(0,0,1);
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

    if (pts.size() >= 2)
        cv::line(img, pts[0], pts[1], cv::Scalar(0,0,255), 2, cv::LINE_AA);
}

void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
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
