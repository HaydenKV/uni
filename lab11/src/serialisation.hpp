#ifndef SERIALISATION_HPP
#define SERIALISATION_HPP

#include <string>
#include <opencv2/core/persistence.hpp>

// Helper templates for OpenCV serialisation

template <typename T>
void read(const cv::FileNode & node, T & obj, const T & default_value = T())
{
    if (node.empty())
        obj = default_value;
    else
        obj.read(node);
}

template <typename T>
void write(cv::FileStorage & fs, const std::string &, const T & obj)
{
    obj.write(fs);
}

#endif