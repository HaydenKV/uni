#include <doctest/doctest.h>
#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>
#include "../../src/DuckDetectorONNX.h"

// Test-only class to exercise DuckDetectorBase without loading ONNX.
class TestDuckDetector : public DuckDetectorBase {
public:
    cv::Mat detect(const cv::Mat & image) override { return image; }
};

// -------------------------------------------------------------------------------------------------
// DuckDetector postprocess — empty & low-score
// Purpose:
//  • testing that empty tensors and low duck-class scores yield no detections.
// -------------------------------------------------------------------------------------------------
SCENARIO("DuckDetectorBase postprocess — empty and low-score")
{
    GIVEN("A detector and an output image")
    {
        TestDuckDetector detector;
        cv::Mat imgout = cv::Mat::zeros(512, 512, CV_8UC3);

        WHEN("Inputs are empty")
        {
            detector.postprocess({}, {}, {1,0,2}, {1,0,512,512}, imgout);
            CHECK(detector.last_centroids().empty());
            CHECK(detector.last_areas().empty());
        }

        WHEN("Duck score is low")
        {
            std::vector<float> class_scores = {0.6f, 0.4f};
            std::vector<float> mask_probs(512*512, 0.6f);
            detector.postprocess(class_scores, mask_probs, {1,1,2}, {1,1,512,512}, imgout);
            CHECK(detector.last_centroids().empty());
            CHECK(detector.last_areas().empty());
        }
    }
}  // derived from base tests :contentReference[oaicite:7]{index=7}

// -------------------------------------------------------------------------------------------------
// DuckDetector postprocess — valid single detection
// Purpose:
//  • testing that centroid and area are computed for a confident square mask.
// -------------------------------------------------------------------------------------------------
SCENARIO("DuckDetectorBase postprocess — valid mask → centroid/area")
{
    GIVEN("A detector and one confident square mask")
    {
        TestDuckDetector detector;
        cv::Mat imgout = cv::Mat::zeros(512, 512, CV_8UC3);

        // 100×100 square centered at ~ (249.5, 249.5)
        std::vector<float> class_scores = {0.4f, 0.6f};
        std::vector<float> mask_probs(512*512, 0.0f);
        for (int i = 200; i < 300; ++i)
            for (int j = 200; j < 300; ++j)
                mask_probs[i*512 + j] = 0.7f;

        detector.postprocess(class_scores, mask_probs, {1,1,2}, {1,1,512,512}, imgout);

        THEN("One centroid/area is produced")
        {
            REQUIRE(detector.last_centroids().size() == 1);
            REQUIRE(detector.last_areas().size() == 1);
            CHECK(detector.last_centroids()[0].x == doctest::Approx(249.5));
            CHECK(detector.last_centroids()[0].y == doctest::Approx(249.5));
            CHECK(detector.last_areas()[0] == doctest::Approx(10000));
        }
    }
}  // preserved from working example :contentReference[oaicite:8]{index=8}

// -------------------------------------------------------------------------------------------------
// DuckDetector postprocess — area gates
// Purpose:
//  • testing for rejection of specks (too small) and floods (too large).
// -------------------------------------------------------------------------------------------------
SCENARIO("DuckDetectorBase postprocess — area gates")
{
    GIVEN("A detector and a blank canvas")
    {
        TestDuckDetector det;
        cv::Mat imgout = cv::Mat::zeros(512, 512, CV_8UC3);

        auto run_square_mask = [&](int box_size_px, float prob, float duck_score)
        {
            const int N = 512;
            std::vector<float> class_scores = {1.0f - duck_score, duck_score};
            std::vector<float> mask_probs   = std::vector<float>(N*N, 0.0f);

            const int s = box_size_px;
            const int start = (N - s)/2;
            const int end   = start + s;
            for (int r = start; r < end; ++r)
                for (int c = start; c < end; ++c)
                    mask_probs[r*N + c] = prob;

            det.postprocess(class_scores, mask_probs, {1,1,2}, {1,1,N,N}, imgout);
            return std::make_pair(det.last_centroids().size(), det.last_areas().size());
        };

        WHEN("A tiny speck is present")
        {
            auto [nc, na] = run_square_mask(/*box*/3, /*prob*/0.9f, /*duck*/0.95f);
            CHECK(nc == 0);
            CHECK(na == 0);
        }

        WHEN("A near-flood covers most of the frame")
        {
            auto [nc, na] = run_square_mask(/*box*/480, /*prob*/0.9f, /*duck*/0.95f);
            CHECK(nc == 0);
            CHECK(na == 0);
        }
    }
}  // area-gate checks merged here :contentReference[oaicite:9]{index=9}
