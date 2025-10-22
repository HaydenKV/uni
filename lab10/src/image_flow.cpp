#include <filesystem>
#include <string>
#include <print>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>

#include "BufferedVideo.h"
#include "Camera.h"
#include "image_flow.h"

// -----------------------------------------------------------------------------
// Helper: (Re)seed features on the (k-1) frame WITHOUT replacing existing tracks
// Adds up to (maxNumFeatures - current) new corners, using a suppression mask
// around existing points to avoid duplicates in a small neighborhood.
// -----------------------------------------------------------------------------
static void reseedOnKm1(const cv::Mat& gray_km1,
                        std::vector<cv::Point2f>& pts_km1,
                        int maxNumFeatures)
{
    const int current = static_cast<int>(pts_km1.size());
    const int slots   = std::max(0, maxNumFeatures - current);
    if (slots <= 0) return;

    // Suppression mask: block neighborhoods around existing points so we don't
    // re-detect right on top of a track we already have.
    cv::Mat mask(gray_km1.size(), CV_8UC1, cv::Scalar(255));
    const int blockRadiusPx = 6; // match/relate to minDistance used in gFTT
    for (const auto& p : pts_km1)
    {
        if (0 <= p.x && p.x < gray_km1.cols && 0 <= p.y && p.y < gray_km1.rows)
            cv::circle(mask, p, blockRadiusPx, cv::Scalar(0), cv::FILLED);
    }

    std::vector<cv::Point2f> seeds;
    cv::goodFeaturesToTrack(
        gray_km1,          // source image: previous frame (k-1)
        seeds,             // output: initial corners
        slots,             // only fill remaining capacity
        0.01,              // qualityLevel
        6.0,               // minDistance
        mask               // honor suppression around existing points
    );

    if (!seeds.empty())
    {
        cv::cornerSubPix(
            gray_km1,
            seeds,
            cv::Size(11,11),
            cv::Size(-1,-1),
            cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT, 30, 1e-3)
        );
        // Append (do NOT replace): keep survivors + add new ones for smoothness
        pts_km1.insert(pts_km1.end(), seeds.begin(), seeds.end());
    }
}

void runImageFlow(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();

    // TODO: Lab 10
    int divisor         = 2;      // Image scaling factor (1,2,4)
    int imgModulus      = 10;      // Use every Nth frame (1 = all frames) Take frames divisible by this number
    int maxNumFeatures  = 1200;   // Maximum number of features per frame
    int minNumFeatures  = 1000;    // Minimum number of feature per frame Re-seed if tracks fall below this

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

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);

    std::println("Input video: {}", videoPath.string());
    std::println("Total number of frames: {}", nFrames);
    double fps = cap.get(cv::CAP_PROP_FPS);
    assert(fps > 0);
    std::println("Input video frame rate: {}", fps);
    std::println("Input video dimensions: [{} x {}]",
                cap.get(cv::CAP_PROP_FRAME_WIDTH),
                cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::println("{:>20}: {}", "maxNumFeatures", maxNumFeatures);
    std::println("{:>20}: {}", "minNumFeatures", minNumFeatures);
    std::println("{:>20}: {}", "divisor", divisor);
    std::println("{:>20}: {}", "imgModulus", imgModulus);
    std::println("");

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        double outputFps    = fps/imgModulus;
        cv::Size frameSize;
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Image flow
    cv::Mat frameFull, frame, gray_k, gray_km1, vis;
    std::vector<cv::Point2f> pts_km1, pts_k; // Feature points at frames k-1 and k

    bool havePrev   = false;
    bool needReseed = false; // triggers detection at the *start* of the next iteration (on k-1)
    int idxFrame    = 0;

    for (;;)
    {
        // a) Extract images from the input video (via BufferedVideo)                  [Task 2a]
        frameFull = bufferedVideoReader.read();
        if (frameFull.empty()) break; // End of video

        // Respect frame-skipping policy: use only frames divisible by imgModulus
        if ((idxFrame++ % imgModulus) != 0) continue;

        // Optional downscale for performance (tracking + drawing happen at this scale)
        if (divisor > 1) {
            cv::resize(frameFull, frame,
                    cv::Size(frameFull.cols/divisor, frameFull.rows/divisor),
                    0, 0, cv::INTER_AREA);
        } else {
            frame = frameFull;
        }

        // Create grayscale copy of current frame I[k]                                  [Task 2a]
        cv::cvtColor(frame, gray_k, cv::COLOR_BGR2GRAY);

        // First valid frame: initialise I[k-1] and continue to get a next frame
        if (!havePrev)
        {
            gray_km1 = gray_k.clone();
            havePrev = true;

            // Ensure we will have features on the first LK step (seed next loop)
            needReseed = true; // NEW: guarantees initial detection on k-1 before first LK

            // Optional: show/export this first frame
            vis = frame.clone();
            if (doExport) {
                bufferedVideoWriter.write(vis);
            } else {
                cv::imshow("LK Demo", vis);
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') break;
            }
            continue;
        }

        // --- reseed policy -------------------------------------------------------
        // We perform (re)seeding on the (k-1) frame BEFORE computing LK for this pair,
        // but the *decision* to reseed is made at the end of the previous iteration
        // based on the number of surviving associations (post-LK). This keeps the
        // initialization on k-1 and preserves smooth motion into k (lab requirement).
        // Additionally, if we have no features at all (e.g., at startup), seed now.
        if (needReseed || pts_km1.empty())
        {
            reseedOnKm1(gray_km1, pts_km1, maxNumFeatures);
            needReseed = false; // consumed
        }
        // ---------------------------------------------------------------------------

        // c.i) Compute LK flow r_Q/O[k-1] -> r_Q/O[k]                                    [Task 2c-i]
        std::vector<unsigned char> status;
        std::vector<float> err;
        std::vector<cv::Point2f> pts_k_tmp;

        cv::calcOpticalFlowPyrLK(
            gray_km1, gray_k,          // previous and current gray frames
            pts_km1, pts_k_tmp,        // previous points -> estimated current points
            status, err,
            cv::Size(21,21),           // window size
            3,                         // max pyramid levels
            cv::TermCriteria(cv::TermCriteria::EPS|cv::TermCriteria::COUNT, 30, 0.01),
            0,                         // flags
            1e-4                       // minEigThreshold
        );

        // c.ii) Filter by LK status (keep only successful tracks)                       [Task 2c-ii]
        pts_k.clear(); pts_k.reserve(pts_k_tmp.size());
        std::vector<cv::Point2f> pts_km1_f; pts_km1_f.reserve(pts_k_tmp.size());
        for (size_t i = 0; i < pts_k_tmp.size(); ++i) {
            if (status[i]) {
                pts_k.push_back(pts_k_tmp[i]);
                pts_km1_f.push_back(pts_km1[i]);
            }
        }
        pts_km1.swap(pts_km1_f);

        // Task 5(a): undistort r^i_{Q/O}[k] and r^i_{Q/O}[k-1] (account for 'divisor'
        // Camera calibration is at full resolution, so scale up before undistorting
        std::vector<cv::Point2f> und_km1_full_cv, und_k_full_cv; // undistorted (full-res) pixels for gating
        std::vector<uchar> inlierMask; inlierMask.assign(pts_k.size(), 0);

        if (pts_k.size() >= 8) { // need at least 8 points for F with RANSAC to be meaningful
            const int N = static_cast<int>(pts_k.size());

            // Scale downscaled coords → full-resolution coords
            Eigen::Matrix<double,2,Eigen::Dynamic> uv_km1_full(2, N), uv_k_full(2, N);
            for (int i = 0; i < N; ++i) {
                uv_km1_full(0,i) = static_cast<double>(pts_km1[i].x) * divisor;
                uv_km1_full(1,i) = static_cast<double>(pts_km1[i].y) * divisor;
                uv_k_full(0,i)   = static_cast<double>(pts_k[i].x)   * divisor;
                uv_k_full(1,i)   = static_cast<double>(pts_k[i].y)   * divisor;
            }

            // Apply distortion correction: r^i_{Q/O} → r^i_{Q/O}_bar
            Eigen::Matrix<double,2,Eigen::Dynamic> uvbar_km1_full = camera.undistort(uv_km1_full);
            Eigen::Matrix<double,2,Eigen::Dynamic> uvbar_k_full   = camera.undistort(uv_k_full);

            // Convert to OpenCV format
            und_km1_full_cv.reserve(N);
            und_k_full_cv.reserve(N);
            for (int i = 0; i < N; ++i) {
                und_km1_full_cv.emplace_back(
                    static_cast<float>(uvbar_km1_full(0,i)),
                    static_cast<float>(uvbar_km1_full(1,i))
                );
                und_k_full_cv.emplace_back(
                    static_cast<float>(uvbar_k_full(0,i)),
                    static_cast<float>(uvbar_k_full(1,i))
                );
            }

            // Task 5(b): find inliers via cv::findFundamentalMat (RANSAC)
            {
                const double ransacThreshPx = 1.0; // epipolar error threshold in pixels (on undistorted full-res coords)
                const double confidence      = 0.999;
                const int    maxIters        = 500;

                cv::Mat F = cv::findFundamentalMat(
                    und_km1_full_cv,               // points at k-1 (undistorted, full-res)
                    und_k_full_cv,                  // points at k   (undistorted, full-res)
                    cv::FM_RANSAC,
                    ransacThreshPx,
                    confidence,
                    maxIters,
                    inlierMask                      // output mask
                );
                // If estimation fails, leave inlierMask as zeros.
                (void)F;
            }
        }

        // Task 5(c): draw flow vectors (green=inlier, red=outlier)
        vis = frame.clone(); // draw at downscaled resolution for display/export
        for (size_t i = 0; i < pts_km1.size(); ++i) {
            const bool inlier = (i < inlierMask.size() && inlierMask[i] != 0);
            const cv::Scalar col = inlier ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
            cv::arrowedLine(vis, pts_km1[i], pts_k[i], col, 2, cv::LINE_AA, 0, 0.25);
        }

        if (doExport) {
            // Task 5(d): export each processed frame with overlay
            bufferedVideoWriter.write(vis);
        } else {
            cv::imshow("LK Demo", vis);
            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break;
        }

        // c.iv) Roll state forward: I[k] -> I[k-1], r[k] -> r[k-1]                       [Task 2c-iv]
        gray_km1 = gray_k.clone();
        pts_km1  = pts_k;

        // --- reseed decision (post-association) -----------------------------------
        // Decide *now* whether we need to reseed next iteration, based on associations
        // that actually survived LK + status filtering. This implements the lab's
        // "re-initialize when tracks fall below a threshold" requirement correctly.
        const int numAssoc = static_cast<int>(pts_km1.size());
        if (numAssoc < minNumFeatures) {
            needReseed = true; // will be executed at the start of the next loop on k-1
        }
        // c.v) Reseed condition is handled at the top of the loop                        [Task 2c-v]
    }

    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();
}
