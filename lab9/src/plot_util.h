#ifndef PLOT_UTIL_H
#define PLOT_UTIL_H

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "image_features.h"

// -------------------------------------------------------
// Function prototypes
// -------------------------------------------------------
void plotGaussianConfidenceEllipse(cv::Mat & img, const GaussianInfo<double> & prQOi, const Eigen::Vector3d & color);
void plotAllFeatures(cv::Mat & img, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y);
void plotMatchedFeatures(cv::Mat & img, const std::vector<int> & idx, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y);
void plotLandmarkIndex(cv::Mat & img, const Eigen::Vector2d & murQOi, const Eigen::Vector3d & colour, int idxLandmark);

#endif
