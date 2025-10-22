#include <string>  
#include <print>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute Harris response
    cv::Mat harrisResponse;
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    cv::cornerHarris(gray, harrisResponse, blockSize, apertureSize, k);

    // --- Normalization removed ---
    // cv::Mat harrisNorm;
    // cv::normalize(harrisResponse, harrisNorm, 0, 255, cv::NORM_MINMAX);
    // harrisNorm.convertTo(harrisNorm, CV_32F);

    // Use raw response instead
    cv::Mat harrisNorm = harrisResponse.clone();

    // Define structure to hold corner data
    struct Corner {
        cv::Point pt;
        float score;
    };
    std::vector<Corner> corners;

    // Threshold and collect high-response points
    float threshold = 0.001f;
    for (int y = 0; y < harrisNorm.rows; ++y) {
        for (int x = 0; x < harrisNorm.cols; ++x) {
            float response = harrisNorm.at<float>(y, x);
            if (response > threshold) {
                corners.push_back({cv::Point(x, y), response});
            }
        }
    }

    // Print metadata for Task 1e
    std::println("Using harris feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", corners.size());

    // Sort corners by descending score
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) {
        return a.score > b.score;
    });

    // Limit to top N features
    int count = std::min(maxNumFeatures, static_cast<int>(corners.size()));

    // Draw all corners as small orange circles
    for (const auto& c : corners) {
        cv::circle(imgout, c.pt, 4, cv::Scalar(0, 165, 255), -1); // Orange filled circles
    }

    // Print and annotate top N corners
    for (int i = 0; i < count; ++i) {
        const auto& c = corners[i];

        std::println("idx: {} at point: ({},{}) Harris Score: {}", i, c.pt.x, c.pt.y, c.score);

        // Draw red circle
        cv::circle(imgout, c.pt, 8, cv::Scalar(0, 0, 255), 1);

        // Draw green index label
        cv::putText(imgout, std::to_string(i), c.pt + cv::Point(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute corner strength using minimum eigenvalue method
    cv::Mat minEigenResponse;
    int blockSize = 2;
    cv::cornerMinEigenVal(gray, minEigenResponse, blockSize);

    // --- Normalization removed ---
    // cv::Mat eigenNorm;
    // cv::normalize(minEigenResponse, eigenNorm, 0, 255, cv::NORM_MINMAX);
    // eigenNorm.convertTo(eigenNorm, CV_32F);

    // Use raw eigenvalue response instead
    cv::Mat eigenNorm = minEigenResponse.clone();

    // Define structure to hold corner data
    struct Corner {
        cv::Point pt;
        float score;
    };
    std::vector<Corner> corners;

    // Threshold response
    float threshold = 0.01f;
    for (int y = 0; y < eigenNorm.rows; ++y) {
        for (int x = 0; x < eigenNorm.cols; ++x) {
            float response = eigenNorm.at<float>(y, x);
            if (response > threshold) {
                corners.push_back({cv::Point(x, y), response});
            }
        }
    }

    // Print info
    std::println("Using Shi-Tomasi feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", corners.size());

    // Sort by score
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) {
        return a.score > b.score;
    });

    int count = std::min(maxNumFeatures, static_cast<int>(corners.size()));

    // Draw all corners (orange)
    for (const auto& c : corners) {
        cv::circle(imgout, c.pt, 3, cv::Scalar(0, 165, 255), -1); // orange filled
    }

    // Draw top N (red + labeled)
    for (int i = 0; i < count; ++i) {
        const auto& c = corners[i];
        std::println("idx: {} at point: ({},{}) Shi Score: {}", i, c.pt.x, c.pt.y, c.score);

        // Red circle for top N
        cv::circle(imgout, c.pt, 6, cv::Scalar(0, 0, 255), 2);

        // Green label
        cv::putText(imgout, std::to_string(i), c.pt + cv::Point(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Detect keypoints using FAST
    std::vector<cv::KeyPoint> keypoints;
    int threshold = 85;
    bool nonmaxSuppression = true; //10001
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression);
    detector->detect(gray, keypoints);

    std::println("Using FAST feature detector");
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", keypoints.size());

    // Sort keypoints by response (descending)
    std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.response > b.response;
    });

    int count = std::min(maxNumFeatures, static_cast<int>(keypoints.size()));

    // Draw all keypoints as small orange dots
    for (const auto& kp : keypoints) {
        cv::circle(imgout, kp.pt, 3, cv::Scalar(0, 165, 255), -1);  // Orange filled
    }

    // Draw top N keypoints with larger red circles and index
    for (int i = 0; i < count; ++i) {
        const auto& kp = keypoints[i];
        std::println("idx: {} at point: ({},{}) Score: {}", i, kp.pt.x, kp.pt.y, kp.response);

        // Red circle
        cv::circle(imgout, kp.pt, 6, cv::Scalar(0, 0, 255), 2);

        // Green index label
        cv::putText(imgout, std::to_string(i), kp.pt + cv::Point2f(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    return imgout;
}

cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // Get dictionary and wrap in a Ptr (required for detectMarkers)
    cv::Ptr<cv::aruco::Dictionary> dictionary = 
        cv::makePtr<cv::aruco::Dictionary>(
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)
        );

    // Containers for detected markers
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;

    // Detect ArUco markers
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds);

    // Draw the detected markers
    if (!markerIds.empty())
    {
        cv::aruco::drawDetectedMarkers(imgout, markerCorners, markerIds);

        std::println("Using aruco feature detector");
        std::println("Image width: {}", img.cols);
        std::println("Image height: {}", img.rows);
        std::println("Detected {} marker(s)", markerIds.size());

        // Pair marker IDs with their corners and sort
        std::vector<std::pair<int, std::vector<cv::Point2f>>> sortedMarkers;
        for (size_t i = 0; i < markerIds.size(); ++i)
        {
            sortedMarkers.emplace_back(markerIds[i], markerCorners[i]);
        }

        std::sort(sortedMarkers.begin(), sortedMarkers.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Print sorted output
        for (const auto& marker : sortedMarkers)
        {
            std::print("ID: {} with corners:", marker.first);
            for (int j = 0; j < 4; ++j)
            {
                cv::Point2f pt = marker.second[j];
                std::print(" ({},{})", static_cast<int>(pt.x), static_cast<int>(pt.y));
            }
            std::print("\n");
        }
    }
    else
    {
        std::println("ArUco: No markers detected.");
    }

    return imgout;
}

