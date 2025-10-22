#include <cassert>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include "GaussianInfo.hpp"
#include "plot_util.h"

void plotGaussianConfidenceEllipse(cv::Mat & img, const GaussianInfo<double> & prQOi, const Eigen::Vector3d & colour)
{
    assert(prQOi.dim() == 2);

    const int markerSize              = 24;
    const int markerThickness         = 2;

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi_ellipse = prQOi.confidenceEllipse(3.0, 100);

    cv::Scalar bgr(colour(2), colour(1), colour(0));

    Eigen::VectorXd murQOi = prQOi.mean();
    cv::drawMarker(img, cv::Point(murQOi(0), murQOi(1)), bgr, cv::MARKER_CROSS, markerSize, markerThickness);

    for (int i = 0; i < rQOi_ellipse.cols() - 1; ++i)
    {
        Eigen::VectorXd rQOi_seg1 = rQOi_ellipse.col(i);
        Eigen::VectorXd rQOi_seg2 = rQOi_ellipse.col(i + 1);

        bool isInWidth1  = 0 <= rQOi_seg1(0) && rQOi_seg1(0) <= img.cols - 1;
        bool isInHeight1 = 0 <= rQOi_seg1(1) && rQOi_seg1(1) <= img.rows - 1;
        
        bool isInWidth2  = 0 <= rQOi_seg2(0) && rQOi_seg2(0) <= img.cols - 1;
        bool isInHeight2 = 0 <= rQOi_seg2(1) && rQOi_seg2(1) <= img.rows - 1;
        bool plotLine    = isInWidth1 && isInHeight1 && isInWidth2 && isInHeight2;
        if (plotLine)
        {
            cv::line(
                img, 
                cv::Point(rQOi_seg1(0), rQOi_seg1(1)),
                cv::Point(rQOi_seg2(0), rQOi_seg2(1)),
                bgr,
                2
            );
        }
    }
}

void plotAllFeatures(cv::Mat & img, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y)
{
    for (std::size_t i = 0; i < Y.cols(); ++i)
    {
        cv::circle(img, cv::Point2f(Y(0, i), Y(1, i)), 2, cv::Scalar(0, 0, 127), 2, 8, 0);
    }
}

void plotMatchedFeatures(cv::Mat & img, const std::vector<int> & idx, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y)
{
    for (std::size_t j = 0; j < idx.size(); ++j)
    {
        int i = idx[j];
        if (i >= 0)
        {
            cv::drawMarker(img, cv::Point2f(Y(0, i), Y(1, i)), cv::Scalar(0, 255, 0), cv::MARKER_TILTED_CROSS, 24, 2);
        }
    }
}

void plotLandmarkIndex(cv::Mat & img, const Eigen::Vector2d & murQOi, const Eigen::Vector3d & colour, int idxLandmark)
{
    cv::putText(
        img,
        std::to_string(idxLandmark),
        cv::Point2f(murQOi(0) + 20, murQOi(1) - 20),
        cv::FONT_HERSHEY_SIMPLEX,
        1,                                              // Font scale
        cv::Scalar(colour(2), colour(1), colour(0)),    // Font colour
        2                                               // Font thickness
    );
}
