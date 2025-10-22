#ifndef IMAGE_FEATURES_H
#define IMAGE_FEATURES_H 

#include <vector>
#include <opencv2/core.hpp>

struct PointFeature
{
    PointFeature();
    PointFeature(const double & score_, const double & x_, const double & y_);
    double score, x, y;
    bool operator<(const PointFeature & other) const;   // used for std::sort
};

std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures);

#endif