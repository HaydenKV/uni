#include <cstdint>
#include <string>
#include <vector>
#include <format> 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "DuckDetectorBase.h"

DuckDetectorBase::~DuckDetectorBase() = default;

void DuckDetectorBase::preprocess(const cv::Mat & img, std::vector<float> & input_tensor_values)
{
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(img_rgb, resized, cv::Size(512, 512));
    cv::Mat float_image;
    resized.convertTo(float_image, CV_32F, 1.0 / 255.0);

    cv::Scalar mean(0.36055567, 0.26455822, 0.1505872);
    cv::Scalar std(0.13891927, 0.10404531, 0.09613165);
    cv::Mat normalized;
    cv::subtract(float_image, mean, normalized);
    cv::divide(normalized, std, normalized);

    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < 512; ++h)
            for (int w = 0; w < 512; ++w)
                input_tensor_values[c * 512 * 512 + h * 512 + w] = normalized.at<cv::Vec3f>(h, w)[c];
}

void DuckDetectorBase::postprocess(const std::vector<float> & class_scores_data, const std::vector<float> & mask_probs_data,
                                   const std::vector<std::int64_t> & class_scores_shape, const std::vector<std::int64_t> & mask_probs_shape,
                                   cv::Mat & imgout)
{
    const int num_queries = static_cast<int>(mask_probs_shape[1]);
    const int mask_height = static_cast<int>(mask_probs_shape[2]);
    const int mask_width  = static_cast<int>(mask_probs_shape[3]);
    const int num_classes = static_cast<int>(class_scores_shape[2]);

    // reset outputs for this frame
    centroids_.clear();
    areas_.clear();

    cv::Mat labelMask = cv::Mat::zeros(imgout.size(), CV_32SC1);
    int label = 1;
    std::vector<int> validLabels;

    for (int query = 0; query < num_queries; ++query)
    {
        const float* query_class_scores = class_scores_data.data() + query * num_classes;
        const int predicted_class = int(std::distance(query_class_scores,
                                           std::max_element(query_class_scores, query_class_scores + num_classes)));
        const float class_score = query_class_scores[predicted_class];

        if (predicted_class == 1 && class_score > 0.5f)
        {
            cv::Mat query_mask(mask_height, mask_width, CV_32F,
                               const_cast<float*>(mask_probs_data.data() + query * mask_height * mask_width));
            cv::Mat resized_query_mask;
            cv::resize(query_mask, resized_query_mask, imgout.size(), 0, 0, cv::INTER_LINEAR);

            cv::Mat binary_mask;
            cv::threshold(resized_query_mask, binary_mask, 0.5, 1, cv::THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8U);

            labelMask.setTo(label, binary_mask);
            validLabels.push_back(label);

            // centroid + area
            cv::Moments m = cv::moments(binary_mask, true);
            if (m.m00 > 0.0)
            {
                cv::Point2f c(float(m.m10 / m.m00), float(m.m01 / m.m00));
                centroids_.push_back(c);
                areas_.push_back(static_cast<double>(cv::countNonZero(binary_mask)));
            }

            ++label;
        }
    }

    // Generate unique colors for each label
    std::vector<cv::Vec3b> colorMap(label);
    for (int i = 0; i < static_cast<int>(validLabels.size()); ++i)
    {
        cv::Mat color(1, 1, CV_8UC3);
        color.at<cv::Vec3b>(0, 0) = cv::Vec3b(180 * i / std::max(1,int(validLabels.size())), 255, 255);
        cv::cvtColor(color, color, cv::COLOR_HSV2BGR);
        colorMap[validLabels[i]] = color.at<cv::Vec3b>(0, 0);
    }

    // Color each pixel according to its label
    for (int y = 0; y < imgout.rows; ++y)
        for (int x = 0; x < imgout.cols; ++x)
        {
            int pixelLabel = labelMask.at<int>(y, x);
            if (pixelLabel > 0)
                imgout.at<cv::Vec3b>(y, x) = colorMap[pixelLabel];
        }

    // Draw centroids and labels onto imgout
    for (size_t i = 0; i < centroids_.size(); ++i)
    {
        cv::circle(imgout, centroids_[i], 5, cv::Scalar(255, 255, 255), -1);
        std::string txt = std::format("A={}", static_cast<int>(areas_[i]));
        cv::putText(imgout, txt, centroids_[i] + cv::Point2f(5.f, -5.f),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

