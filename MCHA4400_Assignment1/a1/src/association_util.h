#ifndef ASSOCIATION_UTIL_H
#define ASSOCIATION_UTIL_H

#include <cstddef>
#include <Eigen/Core>
#include <vector>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"
#include "Camera.h"

double snn(const SystemSLAM & system, const GaussianInfo<double> & featureBundleDensity, const std::vector<std::size_t> & idxLandmarks, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera, std::vector<int> & idxFeatures, bool enforceJointCompatibility = false);
bool individualCompatibility(const int & i, const int & j, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const GaussianInfo<double> & density, const double & nSigma);
bool individualCompatibility(const Eigen::Vector2d & y, const GaussianInfo<double> & marginal, const double & nSigma);
bool jointCompatibility(const std::vector<int> & idx, const double & sU, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const GaussianInfo<double> & density, const double & nSigma, double & surprisal);

#endif