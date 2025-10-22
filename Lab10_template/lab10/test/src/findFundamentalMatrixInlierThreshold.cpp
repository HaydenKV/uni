#include <doctest/doctest.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <vector>

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("Fundamental matrix fit: All inliers" * doctest::skip())
{
    GIVEN("Point correspondences between two frames")
    {
        // rQOikm1 - 196 elements: 
        std::vector<cv::Point2d> rQOikm1;
        rQOikm1.push_back(cv::Point2d(             48,              72));
        rQOikm1.push_back(cv::Point2d(             48,         215.889));
        rQOikm1.push_back(cv::Point2d(             48,         359.778));
        rQOikm1.push_back(cv::Point2d(             48,         503.667));
        rQOikm1.push_back(cv::Point2d(             48,         647.556));
        rQOikm1.push_back(cv::Point2d(             48,         791.444));
        rQOikm1.push_back(cv::Point2d(             48,         935.333));
        rQOikm1.push_back(cv::Point2d(             48,         1079.22));
        rQOikm1.push_back(cv::Point2d(        143.947,              72));
        rQOikm1.push_back(cv::Point2d(        143.947,         215.889));
        rQOikm1.push_back(cv::Point2d(        143.947,         359.778));
        rQOikm1.push_back(cv::Point2d(        143.947,         503.667));
        rQOikm1.push_back(cv::Point2d(        143.947,         647.556));
        rQOikm1.push_back(cv::Point2d(        143.947,         791.444));
        rQOikm1.push_back(cv::Point2d(        143.947,         935.333));
        rQOikm1.push_back(cv::Point2d(        143.947,         1079.22));
        rQOikm1.push_back(cv::Point2d(        143.947,         1223.11));
        rQOikm1.push_back(cv::Point2d(        143.947,            1367));
        rQOikm1.push_back(cv::Point2d(        239.895,              72));
        rQOikm1.push_back(cv::Point2d(        239.895,         215.889));
        rQOikm1.push_back(cv::Point2d(        239.895,         359.778));
        rQOikm1.push_back(cv::Point2d(        239.895,         503.667));
        rQOikm1.push_back(cv::Point2d(        239.895,         647.556));
        rQOikm1.push_back(cv::Point2d(        239.895,         791.444));
        rQOikm1.push_back(cv::Point2d(        239.895,         935.333));
        rQOikm1.push_back(cv::Point2d(        239.895,         1079.22));
        rQOikm1.push_back(cv::Point2d(        239.895,         1223.11));
        rQOikm1.push_back(cv::Point2d(        239.895,            1367));
        rQOikm1.push_back(cv::Point2d(        335.842,              72));
        rQOikm1.push_back(cv::Point2d(        335.842,         215.889));
        rQOikm1.push_back(cv::Point2d(        335.842,         359.778));
        rQOikm1.push_back(cv::Point2d(        335.842,         503.667));
        rQOikm1.push_back(cv::Point2d(        335.842,         647.556));
        rQOikm1.push_back(cv::Point2d(        335.842,         791.444));
        rQOikm1.push_back(cv::Point2d(        335.842,         935.333));
        rQOikm1.push_back(cv::Point2d(        335.842,         1079.22));
        rQOikm1.push_back(cv::Point2d(        335.842,         1223.11));
        rQOikm1.push_back(cv::Point2d(        335.842,            1367));
        rQOikm1.push_back(cv::Point2d(        431.789,              72));
        rQOikm1.push_back(cv::Point2d(        431.789,         215.889));
        rQOikm1.push_back(cv::Point2d(        431.789,         359.778));
        rQOikm1.push_back(cv::Point2d(        431.789,         503.667));
        rQOikm1.push_back(cv::Point2d(        431.789,         647.556));
        rQOikm1.push_back(cv::Point2d(        431.789,         791.444));
        rQOikm1.push_back(cv::Point2d(        431.789,         935.333));
        rQOikm1.push_back(cv::Point2d(        431.789,         1079.22));
        rQOikm1.push_back(cv::Point2d(        431.789,         1223.11));
        rQOikm1.push_back(cv::Point2d(        431.789,            1367));
        rQOikm1.push_back(cv::Point2d(        527.737,              72));
        rQOikm1.push_back(cv::Point2d(        527.737,         215.889));
        rQOikm1.push_back(cv::Point2d(        527.737,         359.778));
        rQOikm1.push_back(cv::Point2d(        527.737,         503.667));
        rQOikm1.push_back(cv::Point2d(        527.737,         647.556));
        rQOikm1.push_back(cv::Point2d(        527.737,         791.444));
        rQOikm1.push_back(cv::Point2d(        527.737,         935.333));
        rQOikm1.push_back(cv::Point2d(        527.737,         1079.22));
        rQOikm1.push_back(cv::Point2d(        527.737,         1223.11));
        rQOikm1.push_back(cv::Point2d(        527.737,            1367));
        rQOikm1.push_back(cv::Point2d(        623.684,              72));
        rQOikm1.push_back(cv::Point2d(        623.684,         215.889));
        rQOikm1.push_back(cv::Point2d(        623.684,         359.778));
        rQOikm1.push_back(cv::Point2d(        623.684,         503.667));
        rQOikm1.push_back(cv::Point2d(        623.684,         647.556));
        rQOikm1.push_back(cv::Point2d(        623.684,         791.444));
        rQOikm1.push_back(cv::Point2d(        623.684,         935.333));
        rQOikm1.push_back(cv::Point2d(        623.684,         1079.22));
        rQOikm1.push_back(cv::Point2d(        623.684,         1223.11));
        rQOikm1.push_back(cv::Point2d(        623.684,            1367));
        rQOikm1.push_back(cv::Point2d(        719.632,              72));
        rQOikm1.push_back(cv::Point2d(        719.632,         215.889));
        rQOikm1.push_back(cv::Point2d(        719.632,         359.778));
        rQOikm1.push_back(cv::Point2d(        719.632,         503.667));
        rQOikm1.push_back(cv::Point2d(        719.632,         647.556));
        rQOikm1.push_back(cv::Point2d(        719.632,         791.444));
        rQOikm1.push_back(cv::Point2d(        719.632,         935.333));
        rQOikm1.push_back(cv::Point2d(        719.632,         1079.22));
        rQOikm1.push_back(cv::Point2d(        719.632,         1223.11));
        rQOikm1.push_back(cv::Point2d(        719.632,            1367));
        rQOikm1.push_back(cv::Point2d(        815.579,              72));
        rQOikm1.push_back(cv::Point2d(        815.579,         215.889));
        rQOikm1.push_back(cv::Point2d(        815.579,         359.778));
        rQOikm1.push_back(cv::Point2d(        815.579,         503.667));
        rQOikm1.push_back(cv::Point2d(        815.579,         647.556));
        rQOikm1.push_back(cv::Point2d(        815.579,         791.444));
        rQOikm1.push_back(cv::Point2d(        815.579,         935.333));
        rQOikm1.push_back(cv::Point2d(        815.579,         1079.22));
        rQOikm1.push_back(cv::Point2d(        815.579,         1223.11));
        rQOikm1.push_back(cv::Point2d(        815.579,            1367));
        rQOikm1.push_back(cv::Point2d(        911.526,              72));
        rQOikm1.push_back(cv::Point2d(        911.526,         215.889));
        rQOikm1.push_back(cv::Point2d(        911.526,         359.778));
        rQOikm1.push_back(cv::Point2d(        911.526,         503.667));
        rQOikm1.push_back(cv::Point2d(        911.526,         647.556));
        rQOikm1.push_back(cv::Point2d(        911.526,         791.444));
        rQOikm1.push_back(cv::Point2d(        911.526,         935.333));
        rQOikm1.push_back(cv::Point2d(        911.526,         1079.22));
        rQOikm1.push_back(cv::Point2d(        911.526,         1223.11));
        rQOikm1.push_back(cv::Point2d(        911.526,            1367));
        rQOikm1.push_back(cv::Point2d(        1007.47,              72));
        rQOikm1.push_back(cv::Point2d(        1007.47,         215.889));
        rQOikm1.push_back(cv::Point2d(        1007.47,         359.778));
        rQOikm1.push_back(cv::Point2d(        1007.47,         503.667));
        rQOikm1.push_back(cv::Point2d(        1007.47,         647.556));
        rQOikm1.push_back(cv::Point2d(        1007.47,         791.444));
        rQOikm1.push_back(cv::Point2d(        1007.47,         935.333));
        rQOikm1.push_back(cv::Point2d(        1007.47,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1007.47,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1007.47,            1367));
        rQOikm1.push_back(cv::Point2d(        1103.42,              72));
        rQOikm1.push_back(cv::Point2d(        1103.42,         215.889));
        rQOikm1.push_back(cv::Point2d(        1103.42,         359.778));
        rQOikm1.push_back(cv::Point2d(        1103.42,         503.667));
        rQOikm1.push_back(cv::Point2d(        1103.42,         647.556));
        rQOikm1.push_back(cv::Point2d(        1103.42,         791.444));
        rQOikm1.push_back(cv::Point2d(        1103.42,         935.333));
        rQOikm1.push_back(cv::Point2d(        1103.42,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1103.42,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1103.42,            1367));
        rQOikm1.push_back(cv::Point2d(        1199.37,              72));
        rQOikm1.push_back(cv::Point2d(        1199.37,         215.889));
        rQOikm1.push_back(cv::Point2d(        1199.37,         359.778));
        rQOikm1.push_back(cv::Point2d(        1199.37,         503.667));
        rQOikm1.push_back(cv::Point2d(        1199.37,         647.556));
        rQOikm1.push_back(cv::Point2d(        1199.37,         791.444));
        rQOikm1.push_back(cv::Point2d(        1199.37,         935.333));
        rQOikm1.push_back(cv::Point2d(        1199.37,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1199.37,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1199.37,            1367));
        rQOikm1.push_back(cv::Point2d(        1295.32,              72));
        rQOikm1.push_back(cv::Point2d(        1295.32,         215.889));
        rQOikm1.push_back(cv::Point2d(        1295.32,         359.778));
        rQOikm1.push_back(cv::Point2d(        1295.32,         503.667));
        rQOikm1.push_back(cv::Point2d(        1295.32,         647.556));
        rQOikm1.push_back(cv::Point2d(        1295.32,         791.444));
        rQOikm1.push_back(cv::Point2d(        1295.32,         935.333));
        rQOikm1.push_back(cv::Point2d(        1295.32,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1295.32,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1295.32,            1367));
        rQOikm1.push_back(cv::Point2d(        1391.26,              72));
        rQOikm1.push_back(cv::Point2d(        1391.26,         215.889));
        rQOikm1.push_back(cv::Point2d(        1391.26,         359.778));
        rQOikm1.push_back(cv::Point2d(        1391.26,         503.667));
        rQOikm1.push_back(cv::Point2d(        1391.26,         647.556));
        rQOikm1.push_back(cv::Point2d(        1391.26,         791.444));
        rQOikm1.push_back(cv::Point2d(        1391.26,         935.333));
        rQOikm1.push_back(cv::Point2d(        1391.26,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1391.26,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1391.26,            1367));
        rQOikm1.push_back(cv::Point2d(        1487.21,              72));
        rQOikm1.push_back(cv::Point2d(        1487.21,         215.889));
        rQOikm1.push_back(cv::Point2d(        1487.21,         359.778));
        rQOikm1.push_back(cv::Point2d(        1487.21,         503.667));
        rQOikm1.push_back(cv::Point2d(        1487.21,         647.556));
        rQOikm1.push_back(cv::Point2d(        1487.21,         791.444));
        rQOikm1.push_back(cv::Point2d(        1487.21,         935.333));
        rQOikm1.push_back(cv::Point2d(        1487.21,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1487.21,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1487.21,            1367));
        rQOikm1.push_back(cv::Point2d(        1583.16,              72));
        rQOikm1.push_back(cv::Point2d(        1583.16,         215.889));
        rQOikm1.push_back(cv::Point2d(        1583.16,         359.778));
        rQOikm1.push_back(cv::Point2d(        1583.16,         503.667));
        rQOikm1.push_back(cv::Point2d(        1583.16,         647.556));
        rQOikm1.push_back(cv::Point2d(        1583.16,         791.444));
        rQOikm1.push_back(cv::Point2d(        1583.16,         935.333));
        rQOikm1.push_back(cv::Point2d(        1583.16,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1583.16,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1583.16,            1367));
        rQOikm1.push_back(cv::Point2d(        1679.11,              72));
        rQOikm1.push_back(cv::Point2d(        1679.11,         215.889));
        rQOikm1.push_back(cv::Point2d(        1679.11,         359.778));
        rQOikm1.push_back(cv::Point2d(        1679.11,         503.667));
        rQOikm1.push_back(cv::Point2d(        1679.11,         647.556));
        rQOikm1.push_back(cv::Point2d(        1679.11,         791.444));
        rQOikm1.push_back(cv::Point2d(        1679.11,         935.333));
        rQOikm1.push_back(cv::Point2d(        1679.11,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1679.11,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1679.11,            1367));
        rQOikm1.push_back(cv::Point2d(        1775.05,              72));
        rQOikm1.push_back(cv::Point2d(        1775.05,         215.889));
        rQOikm1.push_back(cv::Point2d(        1775.05,         359.778));
        rQOikm1.push_back(cv::Point2d(        1775.05,         503.667));
        rQOikm1.push_back(cv::Point2d(        1775.05,         647.556));
        rQOikm1.push_back(cv::Point2d(        1775.05,         791.444));
        rQOikm1.push_back(cv::Point2d(        1775.05,         935.333));
        rQOikm1.push_back(cv::Point2d(        1775.05,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1775.05,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1775.05,            1367));
        rQOikm1.push_back(cv::Point2d(           1871,              72));
        rQOikm1.push_back(cv::Point2d(           1871,         215.889));
        rQOikm1.push_back(cv::Point2d(           1871,         359.778));
        rQOikm1.push_back(cv::Point2d(           1871,         503.667));
        rQOikm1.push_back(cv::Point2d(           1871,         647.556));
        rQOikm1.push_back(cv::Point2d(           1871,         791.444));
        rQOikm1.push_back(cv::Point2d(           1871,         935.333));
        rQOikm1.push_back(cv::Point2d(           1871,         1079.22));

        // rQOik - 196 elements: 
        std::vector<cv::Point2d> rQOik;
        rQOik.push_back(cv::Point2d(             48,              72));
        rQOik.push_back(cv::Point2d(             48,         215.889));
        rQOik.push_back(cv::Point2d(        47.8148,         360.242));
        rQOik.push_back(cv::Point2d(             48,         503.667));
        rQOik.push_back(cv::Point2d(             48,         647.556));
        rQOik.push_back(cv::Point2d(        40.7141,         791.973));
        rQOik.push_back(cv::Point2d(        24.5319,         940.722));
        rQOik.push_back(cv::Point2d(        7.77232,         1094.78));
        rQOik.push_back(cv::Point2d(        143.947,              72));
        rQOik.push_back(cv::Point2d(        143.947,         215.889));
        rQOik.push_back(cv::Point2d(        143.947,         359.778));
        rQOik.push_back(cv::Point2d(        143.947,         503.667));
        rQOik.push_back(cv::Point2d(        143.947,         647.556));
        rQOik.push_back(cv::Point2d(        137.425,         791.973));
        rQOik.push_back(cv::Point2d(        122.937,         940.722));
        rQOik.push_back(cv::Point2d(        107.933,         1094.78));
        rQOik.push_back(cv::Point2d(        92.3842,         1254.43));
        rQOik.push_back(cv::Point2d(        76.2598,         1419.99));
        rQOik.push_back(cv::Point2d(        239.895,              72));
        rQOik.push_back(cv::Point2d(        239.895,         215.889));
        rQOik.push_back(cv::Point2d(        239.895,         359.778));
        rQOik.push_back(cv::Point2d(        239.895,         503.667));
        rQOik.push_back(cv::Point2d(        239.895,         647.556));
        rQOik.push_back(cv::Point2d(        234.135,         791.973));
        rQOik.push_back(cv::Point2d(        221.343,         940.722));
        rQOik.push_back(cv::Point2d(        208.094,         1094.78));
        rQOik.push_back(cv::Point2d(        194.365,         1254.43));
        rQOik.push_back(cv::Point2d(        180.127,         1419.99));
        rQOik.push_back(cv::Point2d(        335.842,              72));
        rQOik.push_back(cv::Point2d(        335.842,         215.889));
        rQOik.push_back(cv::Point2d(        335.842,         359.778));
        rQOik.push_back(cv::Point2d(        335.842,         503.667));
        rQOik.push_back(cv::Point2d(        335.842,         647.556));
        rQOik.push_back(cv::Point2d(        330.846,         791.973));
        rQOik.push_back(cv::Point2d(        319.749,         940.722));
        rQOik.push_back(cv::Point2d(        308.255,         1094.78));
        rQOik.push_back(cv::Point2d(        296.345,         1254.43));
        rQOik.push_back(cv::Point2d(        283.994,         1419.99));
        rQOik.push_back(cv::Point2d(        431.789,              72));
        rQOik.push_back(cv::Point2d(        431.789,         215.889));
        rQOik.push_back(cv::Point2d(        431.789,         359.778));
        rQOik.push_back(cv::Point2d(        431.789,         503.667));
        rQOik.push_back(cv::Point2d(        431.789,         647.556));
        rQOik.push_back(cv::Point2d(        427.556,         791.973));
        rQOik.push_back(cv::Point2d(        418.154,         940.722));
        rQOik.push_back(cv::Point2d(        408.417,         1094.78));
        rQOik.push_back(cv::Point2d(        398.325,         1254.43));
        rQOik.push_back(cv::Point2d(        387.861,         1419.99));
        rQOik.push_back(cv::Point2d(        527.737,              72));
        rQOik.push_back(cv::Point2d(        527.737,         215.889));
        rQOik.push_back(cv::Point2d(        527.737,         359.778));
        rQOik.push_back(cv::Point2d(        527.737,         503.667));
        rQOik.push_back(cv::Point2d(        527.737,         647.556));
        rQOik.push_back(cv::Point2d(        524.267,         791.973));
        rQOik.push_back(cv::Point2d(         516.56,         940.722));
        rQOik.push_back(cv::Point2d(        508.578,         1094.78));
        rQOik.push_back(cv::Point2d(        500.306,         1254.43));
        rQOik.push_back(cv::Point2d(        491.727,         1419.99));
        rQOik.push_back(cv::Point2d(        623.684,              72));
        rQOik.push_back(cv::Point2d(        623.684,         215.889));
        rQOik.push_back(cv::Point2d(        623.684,         359.778));
        rQOik.push_back(cv::Point2d(        623.684,         503.667));
        rQOik.push_back(cv::Point2d(        623.684,         647.556));
        rQOik.push_back(cv::Point2d(        620.977,         791.973));
        rQOik.push_back(cv::Point2d(        614.965,         940.722));
        rQOik.push_back(cv::Point2d(        608.739,         1094.78));
        rQOik.push_back(cv::Point2d(        602.286,         1254.43));
        rQOik.push_back(cv::Point2d(        595.594,         1419.99));
        rQOik.push_back(cv::Point2d(        719.632,              72));
        rQOik.push_back(cv::Point2d(        719.632,         215.889));
        rQOik.push_back(cv::Point2d(        719.632,         359.778));
        rQOik.push_back(cv::Point2d(        719.632,         503.667));
        rQOik.push_back(cv::Point2d(        719.632,         647.556));
        rQOik.push_back(cv::Point2d(        717.688,         791.973));
        rQOik.push_back(cv::Point2d(        713.371,         940.722));
        rQOik.push_back(cv::Point2d(          708.9,         1094.78));
        rQOik.push_back(cv::Point2d(        704.266,         1254.43));
        rQOik.push_back(cv::Point2d(        699.461,         1419.99));
        rQOik.push_back(cv::Point2d(        815.579,              72));
        rQOik.push_back(cv::Point2d(        815.579,         215.889));
        rQOik.push_back(cv::Point2d(        815.579,         359.778));
        rQOik.push_back(cv::Point2d(        815.579,         503.667));
        rQOik.push_back(cv::Point2d(        815.579,         647.556));
        rQOik.push_back(cv::Point2d(        814.398,         791.973));
        rQOik.push_back(cv::Point2d(        811.776,         940.722));
        rQOik.push_back(cv::Point2d(        809.061,         1094.78));
        rQOik.push_back(cv::Point2d(        806.247,         1254.43));
        rQOik.push_back(cv::Point2d(        803.328,         1419.99));
        rQOik.push_back(cv::Point2d(        911.526,              72));
        rQOik.push_back(cv::Point2d(        911.526,         215.889));
        rQOik.push_back(cv::Point2d(        911.526,         359.778));
        rQOik.push_back(cv::Point2d(        911.526,         503.667));
        rQOik.push_back(cv::Point2d(        911.526,         647.556));
        rQOik.push_back(cv::Point2d(        911.109,         791.973));
        rQOik.push_back(cv::Point2d(        910.182,         940.722));
        rQOik.push_back(cv::Point2d(        909.222,         1094.78));
        rQOik.push_back(cv::Point2d(        908.227,         1254.43));
        rQOik.push_back(cv::Point2d(        907.195,         1419.99));
        rQOik.push_back(cv::Point2d(        1007.47,              72));
        rQOik.push_back(cv::Point2d(        1007.47,         215.889));
        rQOik.push_back(cv::Point2d(        1007.47,         359.778));
        rQOik.push_back(cv::Point2d(        1007.47,         503.667));
        rQOik.push_back(cv::Point2d(        1007.47,         647.556));
        rQOik.push_back(cv::Point2d(        1007.82,         791.973));
        rQOik.push_back(cv::Point2d(        1008.59,         940.722));
        rQOik.push_back(cv::Point2d(        1009.38,         1094.78));
        rQOik.push_back(cv::Point2d(        1010.21,         1254.43));
        rQOik.push_back(cv::Point2d(        1011.06,         1419.99));
        rQOik.push_back(cv::Point2d(        1103.42,              72));
        rQOik.push_back(cv::Point2d(        1103.42,         215.889));
        rQOik.push_back(cv::Point2d(        1103.42,         359.778));
        rQOik.push_back(cv::Point2d(        1103.42,         503.667));
        rQOik.push_back(cv::Point2d(        1103.42,         647.556));
        rQOik.push_back(cv::Point2d(        1104.53,         791.973));
        rQOik.push_back(cv::Point2d(        1106.99,         940.722));
        rQOik.push_back(cv::Point2d(        1109.54,         1094.78));
        rQOik.push_back(cv::Point2d(        1112.19,         1254.43));
        rQOik.push_back(cv::Point2d(        1114.93,         1419.99));
        rQOik.push_back(cv::Point2d(        1199.37,              72));
        rQOik.push_back(cv::Point2d(        1199.37,         215.889));
        rQOik.push_back(cv::Point2d(        1199.37,         359.778));
        rQOik.push_back(cv::Point2d(        1199.37,         503.667));
        rQOik.push_back(cv::Point2d(        1199.37,         647.556));
        rQOik.push_back(cv::Point2d(        1201.24,         791.973));
        rQOik.push_back(cv::Point2d(         1205.4,         940.722));
        rQOik.push_back(cv::Point2d(        1209.71,         1094.78));
        rQOik.push_back(cv::Point2d(        1214.17,         1254.43));
        rQOik.push_back(cv::Point2d(         1218.8,         1419.99));
        rQOik.push_back(cv::Point2d(        1295.32,              72));
        rQOik.push_back(cv::Point2d(        1295.32,         215.889));
        rQOik.push_back(cv::Point2d(        1295.32,         359.778));
        rQOik.push_back(cv::Point2d(        1295.32,         503.667));
        rQOik.push_back(cv::Point2d(        1295.32,         647.556));
        rQOik.push_back(cv::Point2d(        1297.95,         791.973));
        rQOik.push_back(cv::Point2d(         1303.8,         940.722));
        rQOik.push_back(cv::Point2d(        1309.87,         1094.78));
        rQOik.push_back(cv::Point2d(        1316.15,         1254.43));
        rQOik.push_back(cv::Point2d(        1322.66,         1419.99));
        rQOik.push_back(cv::Point2d(        1391.26,              72));
        rQOik.push_back(cv::Point2d(        1391.26,         215.889));
        rQOik.push_back(cv::Point2d(        1391.26,         359.778));
        rQOik.push_back(cv::Point2d(        1391.26,         503.667));
        rQOik.push_back(cv::Point2d(        1391.26,         647.556));
        rQOik.push_back(cv::Point2d(        1394.66,         791.973));
        rQOik.push_back(cv::Point2d(        1402.21,         940.722));
        rQOik.push_back(cv::Point2d(        1410.03,         1094.78));
        rQOik.push_back(cv::Point2d(        1418.13,         1254.43));
        rQOik.push_back(cv::Point2d(        1426.53,         1419.99));
        rQOik.push_back(cv::Point2d(        1487.21,              72));
        rQOik.push_back(cv::Point2d(        1487.21,         215.889));
        rQOik.push_back(cv::Point2d(        1487.21,         359.778));
        rQOik.push_back(cv::Point2d(        1487.21,         503.667));
        rQOik.push_back(cv::Point2d(        1487.21,         647.556));
        rQOik.push_back(cv::Point2d(        1491.37,         791.973));
        rQOik.push_back(cv::Point2d(        1500.62,         940.722));
        rQOik.push_back(cv::Point2d(        1510.19,         1094.78));
        rQOik.push_back(cv::Point2d(        1520.11,         1254.43));
        rQOik.push_back(cv::Point2d(         1530.4,         1419.99));
        rQOik.push_back(cv::Point2d(        1583.16,              72));
        rQOik.push_back(cv::Point2d(        1583.16,         215.889));
        rQOik.push_back(cv::Point2d(        1583.16,         359.778));
        rQOik.push_back(cv::Point2d(        1583.16,         503.667));
        rQOik.push_back(cv::Point2d(        1583.16,         647.556));
        rQOik.push_back(cv::Point2d(        1588.08,         791.973));
        rQOik.push_back(cv::Point2d(        1599.02,         940.722));
        rQOik.push_back(cv::Point2d(        1610.35,         1094.78));
        rQOik.push_back(cv::Point2d(        1622.09,         1254.43));
        rQOik.push_back(cv::Point2d(        1634.26,         1419.99));
        rQOik.push_back(cv::Point2d(        1679.11,              72));
        rQOik.push_back(cv::Point2d(        1679.11,         215.889));
        rQOik.push_back(cv::Point2d(        1679.11,         359.778));
        rQOik.push_back(cv::Point2d(        1679.11,         503.667));
        rQOik.push_back(cv::Point2d(        1679.11,         647.556));
        rQOik.push_back(cv::Point2d(        1684.79,         791.973));
        rQOik.push_back(cv::Point2d(        1697.43,         940.722));
        rQOik.push_back(cv::Point2d(        1710.51,         1094.78));
        rQOik.push_back(cv::Point2d(        1724.07,         1254.43));
        rQOik.push_back(cv::Point2d(        1738.13,         1419.99));
        rQOik.push_back(cv::Point2d(        1775.05,              72));
        rQOik.push_back(cv::Point2d(        1775.05,         215.889));
        rQOik.push_back(cv::Point2d(        1775.05,         359.778));
        rQOik.push_back(cv::Point2d(        1775.05,         503.667));
        rQOik.push_back(cv::Point2d(        1775.05,         647.556));
        rQOik.push_back(cv::Point2d(         1781.5,         791.973));
        rQOik.push_back(cv::Point2d(        1795.83,         940.722));
        rQOik.push_back(cv::Point2d(        1810.67,         1094.78));
        rQOik.push_back(cv::Point2d(        1826.05,         1254.43));
        rQOik.push_back(cv::Point2d(           1842,         1419.99));
        rQOik.push_back(cv::Point2d(           1871,              72));
        rQOik.push_back(cv::Point2d(           1871,         215.889));
        rQOik.push_back(cv::Point2d(           1871,         359.778));
        rQOik.push_back(cv::Point2d(           1871,         503.667));
        rQOik.push_back(cv::Point2d(           1871,         647.556));
        rQOik.push_back(cv::Point2d(        1878.21,         791.973));
        rQOik.push_back(cv::Point2d(        1894.24,         940.722));
        rQOik.push_back(cv::Point2d(        1910.83,         1094.78));

        std::vector<uchar> mask;

        WHEN("Calling findFundamentalMat")
        {
            double threshold        = 1; // Epipolar error threshold measured in pixels

            // Call findFundamentalMat
            cv::Mat Fkkm1_cv;
            // TODO: Lab 10

            // Check dimensions
            REQUIRE(Fkkm1_cv.type() == CV_64F);
            REQUIRE(Fkkm1_cv.rows == 3);
            REQUIRE(Fkkm1_cv.cols == 3);

            // Get pixel locations in homogeneous coordinates
            Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1;
            Eigen::Matrix<double, 3, Eigen::Dynamic> pk;
            // TODO: Lab 10

            // Interpret fundamental matrix as an Eigen matrix
            Eigen::Map<Eigen::Matrix3d, Eigen::Unaligned, Eigen::Stride<1, 3>> Fkkm1(Fkkm1_cv.ptr<double>(), 3, 3);

            // Calculate normalised epipolar lines
            Eigen::Matrix<double, 3, Eigen::Dynamic> nlk;
            // TODO: Lab 10

            // Calculate epipolar error
            Eigen::RowVectorXd d;
            // TODO: Lab 10

            THEN("Fundamental matrix has expected properties")
            {
                CAPTURE_EIGEN(Fkkm1);
                REQUIRE(Fkkm1.rows() == 3);
                REQUIRE(Fkkm1.cols() == 3);
                REQUIRE(std::abs(Fkkm1.determinant()) < 1e-8);
                
                // Check Fkkm1(:,1)
                CHECK(Fkkm1(0,0) == doctest::Approx(Fkkm1_cv.at<double>(0,0)));
                CHECK(Fkkm1(1,0) == doctest::Approx(Fkkm1_cv.at<double>(1,0)));
                CHECK(Fkkm1(2,0) == doctest::Approx(Fkkm1_cv.at<double>(2,0)));

                // Check Fkkm1(:,2)
                CHECK(Fkkm1(0,1) == doctest::Approx(Fkkm1_cv.at<double>(0,1)));
                CHECK(Fkkm1(1,1) == doctest::Approx(Fkkm1_cv.at<double>(1,1)));
                CHECK(Fkkm1(2,1) == doctest::Approx(Fkkm1_cv.at<double>(2,1)));

                // Check Fkkm1(:,3)
                CHECK(Fkkm1(0,2) == doctest::Approx(Fkkm1_cv.at<double>(0,2)));
                CHECK(Fkkm1(1,2) == doctest::Approx(Fkkm1_cv.at<double>(1,2)));
                CHECK(Fkkm1(2,2) == doctest::Approx(Fkkm1_cv.at<double>(2,2)));
            }

            THEN("Epipolar error is correct")
            {
                //--------------------------------------------------------------------------------
                // Checks for d 
                //--------------------------------------------------------------------------------
                THEN("d is not empty")
                {
                    REQUIRE(d.size()>0);
                    
                    AND_THEN("d has the right dimensions")
                    {
                        REQUIRE(d.rows()==1);
                        REQUIRE(d.cols()==196);
                        AND_THEN("d is correct")
                        {
                            CHECK(std::abs(std::abs(d(0)) -   1.38555833473e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(1)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(2)) -                 0.5) < 0.012);
                            CHECK(std::abs(std::abs(d(3)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(4)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(5)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(6)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(7)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(8)) -   4.26325641456e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(9)) -   8.52651282912e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(10)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(11)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(12)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(13)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(14)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(15)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(16)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(17)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(18)) -   9.94759830064e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(19)) -   7.81597009336e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(20)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(21)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(22)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(23)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(24)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(25)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(26)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(27)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(28)) -   1.42108547152e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(29)) -   8.52651282912e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(30)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(31)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(32)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(33)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(34)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(35)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(36)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(37)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(38)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(39)) -   1.42108547152e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(40)) -   1.84741111298e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(41)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(42)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(43)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(44)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(45)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(46)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(47)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(48)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(49)) -   5.68434188608e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(50)) -   2.20268248086e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(51)) -   1.42108547152e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(52)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(53)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(54)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(55)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(56)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(57)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(58)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(59)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(60)) -   1.98951966013e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(61)) -   3.12638803734e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(62)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(63)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(64)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(65)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(66)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(67)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(68)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(69)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(70)) -   2.84217094304e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(71)) -   5.40012479178e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(72)) -    6.8212102633e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(73)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(74)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(75)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(76)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(77)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(78)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(79)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(80)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(81)) -   3.97903932026e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(82)) -   7.38964445191e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(83)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(84)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(85)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(86)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(87)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(88)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(89)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(90)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(91)) -   5.68434188608e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(92)) -   1.53477230924e-12) < 0.012);
                            CHECK(std::abs(std::abs(d(93)) -   9.09494701773e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(94)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(95)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(96)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(97)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(98)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(99)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(100)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(101)) -    6.8212102633e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(102)) -   2.27373675443e-12) < 0.012);
                            CHECK(std::abs(std::abs(d(103)) -   1.70530256582e-12) < 0.012);
                            CHECK(std::abs(std::abs(d(104)) -    6.8212102633e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(105)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(106)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(107)) -    6.8212102633e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(108)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(109)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(110)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(111)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(112)) -   9.09494701773e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(113)) -   4.26325641456e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(114)) -   2.84217094304e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(115)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(116)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(117)) -   5.68434188608e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(118)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(119)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(120)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(121)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(122)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(123)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(124)) -   8.52651282912e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(125)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(126)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(127)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(128)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(129)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(130)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(131)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(132)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(133)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(134)) -   5.68434188608e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(135)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(136)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(137)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(138)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(139)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(140)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(141)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(142)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(143)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(144)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(145)) -   3.48165940522e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(146)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(147)) -   3.97903932026e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(148)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(149)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(150)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(151)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(152)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(153)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(154)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(155)) -    1.4921397451e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(156)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(157)) -   5.11590769747e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(158)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(159)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(160)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(161)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(162)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(163)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(164)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(165)) -   2.55795384874e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(166)) -   2.48689957516e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(167)) -   2.84217094304e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(168)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(169)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(170)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(171)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(172)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(173)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(174)) -   5.68434188608e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(175)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(176)) -   2.84217094304e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(177)) -   5.25801624462e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(178)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(179)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(180)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(181)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(182)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(183)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(184)) -   5.68434188608e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(185)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(186)) -   1.84741111298e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(187)) -   3.62376795238e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(188)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(189)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(190)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(191)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(192)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(193)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(194)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(195)) -   3.41060513165e-13) < 0.012);
                        }
                    }
                }
            
            }

            THEN("mask has the correct dimensions")
            {
                REQUIRE(mask.size() == 196);

                AND_THEN("mask has the correct number of inliers")
                {
                    int nInliers = 0;
                    for (int i = 0; i < 196; ++i)
                    {
                        if(mask[i]){
                            nInliers++;
                        }
                    }
                    REQUIRE(nInliers == 196);
                }
            }
        }
    }
}

SCENARIO("Fundamental matrix fit: One outlier, with epipolar error of 2 pixels" * doctest::skip())
{
    GIVEN("Point correspondences between two frames")
    {
        // rQOikm1 - 196 elements: 
        std::vector<cv::Point2d> rQOikm1;
        rQOikm1.push_back(cv::Point2d(             48,              72));
        rQOikm1.push_back(cv::Point2d(             48,         215.889));
        rQOikm1.push_back(cv::Point2d(             48,         359.778));
        rQOikm1.push_back(cv::Point2d(             48,         503.667));
        rQOikm1.push_back(cv::Point2d(             48,         647.556));
        rQOikm1.push_back(cv::Point2d(             48,         791.444));
        rQOikm1.push_back(cv::Point2d(             48,         935.333));
        rQOikm1.push_back(cv::Point2d(             48,         1079.22));
        rQOikm1.push_back(cv::Point2d(        143.947,              72));
        rQOikm1.push_back(cv::Point2d(        143.947,         215.889));
        rQOikm1.push_back(cv::Point2d(        143.947,         359.778));
        rQOikm1.push_back(cv::Point2d(        143.947,         503.667));
        rQOikm1.push_back(cv::Point2d(        143.947,         647.556));
        rQOikm1.push_back(cv::Point2d(        143.947,         791.444));
        rQOikm1.push_back(cv::Point2d(        143.947,         935.333));
        rQOikm1.push_back(cv::Point2d(        143.947,         1079.22));
        rQOikm1.push_back(cv::Point2d(        143.947,         1223.11));
        rQOikm1.push_back(cv::Point2d(        143.947,            1367));
        rQOikm1.push_back(cv::Point2d(        239.895,              72));
        rQOikm1.push_back(cv::Point2d(        239.895,         215.889));
        rQOikm1.push_back(cv::Point2d(        239.895,         359.778));
        rQOikm1.push_back(cv::Point2d(        239.895,         503.667));
        rQOikm1.push_back(cv::Point2d(        239.895,         647.556));
        rQOikm1.push_back(cv::Point2d(        239.895,         791.444));
        rQOikm1.push_back(cv::Point2d(        239.895,         935.333));
        rQOikm1.push_back(cv::Point2d(        239.895,         1079.22));
        rQOikm1.push_back(cv::Point2d(        239.895,         1223.11));
        rQOikm1.push_back(cv::Point2d(        239.895,            1367));
        rQOikm1.push_back(cv::Point2d(        335.842,              72));
        rQOikm1.push_back(cv::Point2d(        335.842,         215.889));
        rQOikm1.push_back(cv::Point2d(        335.842,         359.778));
        rQOikm1.push_back(cv::Point2d(        335.842,         503.667));
        rQOikm1.push_back(cv::Point2d(        335.842,         647.556));
        rQOikm1.push_back(cv::Point2d(        335.842,         791.444));
        rQOikm1.push_back(cv::Point2d(        335.842,         935.333));
        rQOikm1.push_back(cv::Point2d(        335.842,         1079.22));
        rQOikm1.push_back(cv::Point2d(        335.842,         1223.11));
        rQOikm1.push_back(cv::Point2d(        335.842,            1367));
        rQOikm1.push_back(cv::Point2d(        431.789,              72));
        rQOikm1.push_back(cv::Point2d(        431.789,         215.889));
        rQOikm1.push_back(cv::Point2d(        431.789,         359.778));
        rQOikm1.push_back(cv::Point2d(        431.789,         503.667));
        rQOikm1.push_back(cv::Point2d(        431.789,         647.556));
        rQOikm1.push_back(cv::Point2d(        431.789,         791.444));
        rQOikm1.push_back(cv::Point2d(        431.789,         935.333));
        rQOikm1.push_back(cv::Point2d(        431.789,         1079.22));
        rQOikm1.push_back(cv::Point2d(        431.789,         1223.11));
        rQOikm1.push_back(cv::Point2d(        431.789,            1367));
        rQOikm1.push_back(cv::Point2d(        527.737,              72));
        rQOikm1.push_back(cv::Point2d(        527.737,         215.889));
        rQOikm1.push_back(cv::Point2d(        527.737,         359.778));
        rQOikm1.push_back(cv::Point2d(        527.737,         503.667));
        rQOikm1.push_back(cv::Point2d(        527.737,         647.556));
        rQOikm1.push_back(cv::Point2d(        527.737,         791.444));
        rQOikm1.push_back(cv::Point2d(        527.737,         935.333));
        rQOikm1.push_back(cv::Point2d(        527.737,         1079.22));
        rQOikm1.push_back(cv::Point2d(        527.737,         1223.11));
        rQOikm1.push_back(cv::Point2d(        527.737,            1367));
        rQOikm1.push_back(cv::Point2d(        623.684,              72));
        rQOikm1.push_back(cv::Point2d(        623.684,         215.889));
        rQOikm1.push_back(cv::Point2d(        623.684,         359.778));
        rQOikm1.push_back(cv::Point2d(        623.684,         503.667));
        rQOikm1.push_back(cv::Point2d(        623.684,         647.556));
        rQOikm1.push_back(cv::Point2d(        623.684,         791.444));
        rQOikm1.push_back(cv::Point2d(        623.684,         935.333));
        rQOikm1.push_back(cv::Point2d(        623.684,         1079.22));
        rQOikm1.push_back(cv::Point2d(        623.684,         1223.11));
        rQOikm1.push_back(cv::Point2d(        623.684,            1367));
        rQOikm1.push_back(cv::Point2d(        719.632,              72));
        rQOikm1.push_back(cv::Point2d(        719.632,         215.889));
        rQOikm1.push_back(cv::Point2d(        719.632,         359.778));
        rQOikm1.push_back(cv::Point2d(        719.632,         503.667));
        rQOikm1.push_back(cv::Point2d(        719.632,         647.556));
        rQOikm1.push_back(cv::Point2d(        719.632,         791.444));
        rQOikm1.push_back(cv::Point2d(        719.632,         935.333));
        rQOikm1.push_back(cv::Point2d(        719.632,         1079.22));
        rQOikm1.push_back(cv::Point2d(        719.632,         1223.11));
        rQOikm1.push_back(cv::Point2d(        719.632,            1367));
        rQOikm1.push_back(cv::Point2d(        815.579,              72));
        rQOikm1.push_back(cv::Point2d(        815.579,         215.889));
        rQOikm1.push_back(cv::Point2d(        815.579,         359.778));
        rQOikm1.push_back(cv::Point2d(        815.579,         503.667));
        rQOikm1.push_back(cv::Point2d(        815.579,         647.556));
        rQOikm1.push_back(cv::Point2d(        815.579,         791.444));
        rQOikm1.push_back(cv::Point2d(        815.579,         935.333));
        rQOikm1.push_back(cv::Point2d(        815.579,         1079.22));
        rQOikm1.push_back(cv::Point2d(        815.579,         1223.11));
        rQOikm1.push_back(cv::Point2d(        815.579,            1367));
        rQOikm1.push_back(cv::Point2d(        911.526,              72));
        rQOikm1.push_back(cv::Point2d(        911.526,         215.889));
        rQOikm1.push_back(cv::Point2d(        911.526,         359.778));
        rQOikm1.push_back(cv::Point2d(        911.526,         503.667));
        rQOikm1.push_back(cv::Point2d(        911.526,         647.556));
        rQOikm1.push_back(cv::Point2d(        911.526,         791.444));
        rQOikm1.push_back(cv::Point2d(        911.526,         935.333));
        rQOikm1.push_back(cv::Point2d(        911.526,         1079.22));
        rQOikm1.push_back(cv::Point2d(        911.526,         1223.11));
        rQOikm1.push_back(cv::Point2d(        911.526,            1367));
        rQOikm1.push_back(cv::Point2d(        1007.47,              72));
        rQOikm1.push_back(cv::Point2d(        1007.47,         215.889));
        rQOikm1.push_back(cv::Point2d(        1007.47,         359.778));
        rQOikm1.push_back(cv::Point2d(        1007.47,         503.667));
        rQOikm1.push_back(cv::Point2d(        1007.47,         647.556));
        rQOikm1.push_back(cv::Point2d(        1007.47,         791.444));
        rQOikm1.push_back(cv::Point2d(        1007.47,         935.333));
        rQOikm1.push_back(cv::Point2d(        1007.47,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1007.47,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1007.47,            1367));
        rQOikm1.push_back(cv::Point2d(        1103.42,              72));
        rQOikm1.push_back(cv::Point2d(        1103.42,         215.889));
        rQOikm1.push_back(cv::Point2d(        1103.42,         359.778));
        rQOikm1.push_back(cv::Point2d(        1103.42,         503.667));
        rQOikm1.push_back(cv::Point2d(        1103.42,         647.556));
        rQOikm1.push_back(cv::Point2d(        1103.42,         791.444));
        rQOikm1.push_back(cv::Point2d(        1103.42,         935.333));
        rQOikm1.push_back(cv::Point2d(        1103.42,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1103.42,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1103.42,            1367));
        rQOikm1.push_back(cv::Point2d(        1199.37,              72));
        rQOikm1.push_back(cv::Point2d(        1199.37,         215.889));
        rQOikm1.push_back(cv::Point2d(        1199.37,         359.778));
        rQOikm1.push_back(cv::Point2d(        1199.37,         503.667));
        rQOikm1.push_back(cv::Point2d(        1199.37,         647.556));
        rQOikm1.push_back(cv::Point2d(        1199.37,         791.444));
        rQOikm1.push_back(cv::Point2d(        1199.37,         935.333));
        rQOikm1.push_back(cv::Point2d(        1199.37,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1199.37,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1199.37,            1367));
        rQOikm1.push_back(cv::Point2d(        1295.32,              72));
        rQOikm1.push_back(cv::Point2d(        1295.32,         215.889));
        rQOikm1.push_back(cv::Point2d(        1295.32,         359.778));
        rQOikm1.push_back(cv::Point2d(        1295.32,         503.667));
        rQOikm1.push_back(cv::Point2d(        1295.32,         647.556));
        rQOikm1.push_back(cv::Point2d(        1295.32,         791.444));
        rQOikm1.push_back(cv::Point2d(        1295.32,         935.333));
        rQOikm1.push_back(cv::Point2d(        1295.32,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1295.32,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1295.32,            1367));
        rQOikm1.push_back(cv::Point2d(        1391.26,              72));
        rQOikm1.push_back(cv::Point2d(        1391.26,         215.889));
        rQOikm1.push_back(cv::Point2d(        1391.26,         359.778));
        rQOikm1.push_back(cv::Point2d(        1391.26,         503.667));
        rQOikm1.push_back(cv::Point2d(        1391.26,         647.556));
        rQOikm1.push_back(cv::Point2d(        1391.26,         791.444));
        rQOikm1.push_back(cv::Point2d(        1391.26,         935.333));
        rQOikm1.push_back(cv::Point2d(        1391.26,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1391.26,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1391.26,            1367));
        rQOikm1.push_back(cv::Point2d(        1487.21,              72));
        rQOikm1.push_back(cv::Point2d(        1487.21,         215.889));
        rQOikm1.push_back(cv::Point2d(        1487.21,         359.778));
        rQOikm1.push_back(cv::Point2d(        1487.21,         503.667));
        rQOikm1.push_back(cv::Point2d(        1487.21,         647.556));
        rQOikm1.push_back(cv::Point2d(        1487.21,         791.444));
        rQOikm1.push_back(cv::Point2d(        1487.21,         935.333));
        rQOikm1.push_back(cv::Point2d(        1487.21,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1487.21,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1487.21,            1367));
        rQOikm1.push_back(cv::Point2d(        1583.16,              72));
        rQOikm1.push_back(cv::Point2d(        1583.16,         215.889));
        rQOikm1.push_back(cv::Point2d(        1583.16,         359.778));
        rQOikm1.push_back(cv::Point2d(        1583.16,         503.667));
        rQOikm1.push_back(cv::Point2d(        1583.16,         647.556));
        rQOikm1.push_back(cv::Point2d(        1583.16,         791.444));
        rQOikm1.push_back(cv::Point2d(        1583.16,         935.333));
        rQOikm1.push_back(cv::Point2d(        1583.16,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1583.16,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1583.16,            1367));
        rQOikm1.push_back(cv::Point2d(        1679.11,              72));
        rQOikm1.push_back(cv::Point2d(        1679.11,         215.889));
        rQOikm1.push_back(cv::Point2d(        1679.11,         359.778));
        rQOikm1.push_back(cv::Point2d(        1679.11,         503.667));
        rQOikm1.push_back(cv::Point2d(        1679.11,         647.556));
        rQOikm1.push_back(cv::Point2d(        1679.11,         791.444));
        rQOikm1.push_back(cv::Point2d(        1679.11,         935.333));
        rQOikm1.push_back(cv::Point2d(        1679.11,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1679.11,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1679.11,            1367));
        rQOikm1.push_back(cv::Point2d(        1775.05,              72));
        rQOikm1.push_back(cv::Point2d(        1775.05,         215.889));
        rQOikm1.push_back(cv::Point2d(        1775.05,         359.778));
        rQOikm1.push_back(cv::Point2d(        1775.05,         503.667));
        rQOikm1.push_back(cv::Point2d(        1775.05,         647.556));
        rQOikm1.push_back(cv::Point2d(        1775.05,         791.444));
        rQOikm1.push_back(cv::Point2d(        1775.05,         935.333));
        rQOikm1.push_back(cv::Point2d(        1775.05,         1079.22));
        rQOikm1.push_back(cv::Point2d(        1775.05,         1223.11));
        rQOikm1.push_back(cv::Point2d(        1775.05,            1367));
        rQOikm1.push_back(cv::Point2d(           1871,              72));
        rQOikm1.push_back(cv::Point2d(           1871,         215.889));
        rQOikm1.push_back(cv::Point2d(           1871,         359.778));
        rQOikm1.push_back(cv::Point2d(           1871,         503.667));
        rQOikm1.push_back(cv::Point2d(           1871,         647.556));
        rQOikm1.push_back(cv::Point2d(           1871,         791.444));
        rQOikm1.push_back(cv::Point2d(           1871,         935.333));
        rQOikm1.push_back(cv::Point2d(           1871,         1079.22));

        // rQOik - 196 elements: 
        std::vector<cv::Point2d> rQOik;
        rQOik.push_back(cv::Point2d(             48,              72));
        rQOik.push_back(cv::Point2d(             48,         215.889));
        rQOik.push_back(cv::Point2d(        47.2593,         361.636));
        rQOik.push_back(cv::Point2d(             48,         503.667));
        rQOik.push_back(cv::Point2d(             48,         647.556));
        rQOik.push_back(cv::Point2d(        40.7141,         791.973));
        rQOik.push_back(cv::Point2d(        24.5319,         940.722));
        rQOik.push_back(cv::Point2d(        7.77232,         1094.78));
        rQOik.push_back(cv::Point2d(        143.947,              72));
        rQOik.push_back(cv::Point2d(        143.947,         215.889));
        rQOik.push_back(cv::Point2d(        143.947,         359.778));
        rQOik.push_back(cv::Point2d(        143.947,         503.667));
        rQOik.push_back(cv::Point2d(        143.947,         647.556));
        rQOik.push_back(cv::Point2d(        137.425,         791.973));
        rQOik.push_back(cv::Point2d(        122.937,         940.722));
        rQOik.push_back(cv::Point2d(        107.933,         1094.78));
        rQOik.push_back(cv::Point2d(        92.3842,         1254.43));
        rQOik.push_back(cv::Point2d(        76.2598,         1419.99));
        rQOik.push_back(cv::Point2d(        239.895,              72));
        rQOik.push_back(cv::Point2d(        239.895,         215.889));
        rQOik.push_back(cv::Point2d(        239.895,         359.778));
        rQOik.push_back(cv::Point2d(        239.895,         503.667));
        rQOik.push_back(cv::Point2d(        239.895,         647.556));
        rQOik.push_back(cv::Point2d(        234.135,         791.973));
        rQOik.push_back(cv::Point2d(        221.343,         940.722));
        rQOik.push_back(cv::Point2d(        208.094,         1094.78));
        rQOik.push_back(cv::Point2d(        194.365,         1254.43));
        rQOik.push_back(cv::Point2d(        180.127,         1419.99));
        rQOik.push_back(cv::Point2d(        335.842,              72));
        rQOik.push_back(cv::Point2d(        335.842,         215.889));
        rQOik.push_back(cv::Point2d(        335.842,         359.778));
        rQOik.push_back(cv::Point2d(        335.842,         503.667));
        rQOik.push_back(cv::Point2d(        335.842,         647.556));
        rQOik.push_back(cv::Point2d(        330.846,         791.973));
        rQOik.push_back(cv::Point2d(        319.749,         940.722));
        rQOik.push_back(cv::Point2d(        308.255,         1094.78));
        rQOik.push_back(cv::Point2d(        296.345,         1254.43));
        rQOik.push_back(cv::Point2d(        283.994,         1419.99));
        rQOik.push_back(cv::Point2d(        431.789,              72));
        rQOik.push_back(cv::Point2d(        431.789,         215.889));
        rQOik.push_back(cv::Point2d(        431.789,         359.778));
        rQOik.push_back(cv::Point2d(        431.789,         503.667));
        rQOik.push_back(cv::Point2d(        431.789,         647.556));
        rQOik.push_back(cv::Point2d(        427.556,         791.973));
        rQOik.push_back(cv::Point2d(        418.154,         940.722));
        rQOik.push_back(cv::Point2d(        408.417,         1094.78));
        rQOik.push_back(cv::Point2d(        398.325,         1254.43));
        rQOik.push_back(cv::Point2d(        387.861,         1419.99));
        rQOik.push_back(cv::Point2d(        527.737,              72));
        rQOik.push_back(cv::Point2d(        527.737,         215.889));
        rQOik.push_back(cv::Point2d(        527.737,         359.778));
        rQOik.push_back(cv::Point2d(        527.737,         503.667));
        rQOik.push_back(cv::Point2d(        527.737,         647.556));
        rQOik.push_back(cv::Point2d(        524.267,         791.973));
        rQOik.push_back(cv::Point2d(         516.56,         940.722));
        rQOik.push_back(cv::Point2d(        508.578,         1094.78));
        rQOik.push_back(cv::Point2d(        500.306,         1254.43));
        rQOik.push_back(cv::Point2d(        491.727,         1419.99));
        rQOik.push_back(cv::Point2d(        623.684,              72));
        rQOik.push_back(cv::Point2d(        623.684,         215.889));
        rQOik.push_back(cv::Point2d(        623.684,         359.778));
        rQOik.push_back(cv::Point2d(        623.684,         503.667));
        rQOik.push_back(cv::Point2d(        623.684,         647.556));
        rQOik.push_back(cv::Point2d(        620.977,         791.973));
        rQOik.push_back(cv::Point2d(        614.965,         940.722));
        rQOik.push_back(cv::Point2d(        608.739,         1094.78));
        rQOik.push_back(cv::Point2d(        602.286,         1254.43));
        rQOik.push_back(cv::Point2d(        595.594,         1419.99));
        rQOik.push_back(cv::Point2d(        719.632,              72));
        rQOik.push_back(cv::Point2d(        719.632,         215.889));
        rQOik.push_back(cv::Point2d(        719.632,         359.778));
        rQOik.push_back(cv::Point2d(        719.632,         503.667));
        rQOik.push_back(cv::Point2d(        719.632,         647.556));
        rQOik.push_back(cv::Point2d(        717.688,         791.973));
        rQOik.push_back(cv::Point2d(        713.371,         940.722));
        rQOik.push_back(cv::Point2d(          708.9,         1094.78));
        rQOik.push_back(cv::Point2d(        704.266,         1254.43));
        rQOik.push_back(cv::Point2d(        699.461,         1419.99));
        rQOik.push_back(cv::Point2d(        815.579,              72));
        rQOik.push_back(cv::Point2d(        815.579,         215.889));
        rQOik.push_back(cv::Point2d(        815.579,         359.778));
        rQOik.push_back(cv::Point2d(        815.579,         503.667));
        rQOik.push_back(cv::Point2d(        815.579,         647.556));
        rQOik.push_back(cv::Point2d(        814.398,         791.973));
        rQOik.push_back(cv::Point2d(        811.776,         940.722));
        rQOik.push_back(cv::Point2d(        809.061,         1094.78));
        rQOik.push_back(cv::Point2d(        806.247,         1254.43));
        rQOik.push_back(cv::Point2d(        803.328,         1419.99));
        rQOik.push_back(cv::Point2d(        911.526,              72));
        rQOik.push_back(cv::Point2d(        911.526,         215.889));
        rQOik.push_back(cv::Point2d(        911.526,         359.778));
        rQOik.push_back(cv::Point2d(        911.526,         503.667));
        rQOik.push_back(cv::Point2d(        911.526,         647.556));
        rQOik.push_back(cv::Point2d(        911.109,         791.973));
        rQOik.push_back(cv::Point2d(        910.182,         940.722));
        rQOik.push_back(cv::Point2d(        909.222,         1094.78));
        rQOik.push_back(cv::Point2d(        908.227,         1254.43));
        rQOik.push_back(cv::Point2d(        907.195,         1419.99));
        rQOik.push_back(cv::Point2d(        1007.47,              72));
        rQOik.push_back(cv::Point2d(        1007.47,         215.889));
        rQOik.push_back(cv::Point2d(        1007.47,         359.778));
        rQOik.push_back(cv::Point2d(        1007.47,         503.667));
        rQOik.push_back(cv::Point2d(        1007.47,         647.556));
        rQOik.push_back(cv::Point2d(        1007.82,         791.973));
        rQOik.push_back(cv::Point2d(        1008.59,         940.722));
        rQOik.push_back(cv::Point2d(        1009.38,         1094.78));
        rQOik.push_back(cv::Point2d(        1010.21,         1254.43));
        rQOik.push_back(cv::Point2d(        1011.06,         1419.99));
        rQOik.push_back(cv::Point2d(        1103.42,              72));
        rQOik.push_back(cv::Point2d(        1103.42,         215.889));
        rQOik.push_back(cv::Point2d(        1103.42,         359.778));
        rQOik.push_back(cv::Point2d(        1103.42,         503.667));
        rQOik.push_back(cv::Point2d(        1103.42,         647.556));
        rQOik.push_back(cv::Point2d(        1104.53,         791.973));
        rQOik.push_back(cv::Point2d(        1106.99,         940.722));
        rQOik.push_back(cv::Point2d(        1109.54,         1094.78));
        rQOik.push_back(cv::Point2d(        1112.19,         1254.43));
        rQOik.push_back(cv::Point2d(        1114.93,         1419.99));
        rQOik.push_back(cv::Point2d(        1199.37,              72));
        rQOik.push_back(cv::Point2d(        1199.37,         215.889));
        rQOik.push_back(cv::Point2d(        1199.37,         359.778));
        rQOik.push_back(cv::Point2d(        1199.37,         503.667));
        rQOik.push_back(cv::Point2d(        1199.37,         647.556));
        rQOik.push_back(cv::Point2d(        1201.24,         791.973));
        rQOik.push_back(cv::Point2d(         1205.4,         940.722));
        rQOik.push_back(cv::Point2d(        1209.71,         1094.78));
        rQOik.push_back(cv::Point2d(        1214.17,         1254.43));
        rQOik.push_back(cv::Point2d(         1218.8,         1419.99));
        rQOik.push_back(cv::Point2d(        1295.32,              72));
        rQOik.push_back(cv::Point2d(        1295.32,         215.889));
        rQOik.push_back(cv::Point2d(        1295.32,         359.778));
        rQOik.push_back(cv::Point2d(        1295.32,         503.667));
        rQOik.push_back(cv::Point2d(        1295.32,         647.556));
        rQOik.push_back(cv::Point2d(        1297.95,         791.973));
        rQOik.push_back(cv::Point2d(         1303.8,         940.722));
        rQOik.push_back(cv::Point2d(        1309.87,         1094.78));
        rQOik.push_back(cv::Point2d(        1316.15,         1254.43));
        rQOik.push_back(cv::Point2d(        1322.66,         1419.99));
        rQOik.push_back(cv::Point2d(        1391.26,              72));
        rQOik.push_back(cv::Point2d(        1391.26,         215.889));
        rQOik.push_back(cv::Point2d(        1391.26,         359.778));
        rQOik.push_back(cv::Point2d(        1391.26,         503.667));
        rQOik.push_back(cv::Point2d(        1391.26,         647.556));
        rQOik.push_back(cv::Point2d(        1394.66,         791.973));
        rQOik.push_back(cv::Point2d(        1402.21,         940.722));
        rQOik.push_back(cv::Point2d(        1410.03,         1094.78));
        rQOik.push_back(cv::Point2d(        1418.13,         1254.43));
        rQOik.push_back(cv::Point2d(        1426.53,         1419.99));
        rQOik.push_back(cv::Point2d(        1487.21,              72));
        rQOik.push_back(cv::Point2d(        1487.21,         215.889));
        rQOik.push_back(cv::Point2d(        1487.21,         359.778));
        rQOik.push_back(cv::Point2d(        1487.21,         503.667));
        rQOik.push_back(cv::Point2d(        1487.21,         647.556));
        rQOik.push_back(cv::Point2d(        1491.37,         791.973));
        rQOik.push_back(cv::Point2d(        1500.62,         940.722));
        rQOik.push_back(cv::Point2d(        1510.19,         1094.78));
        rQOik.push_back(cv::Point2d(        1520.11,         1254.43));
        rQOik.push_back(cv::Point2d(         1530.4,         1419.99));
        rQOik.push_back(cv::Point2d(        1583.16,              72));
        rQOik.push_back(cv::Point2d(        1583.16,         215.889));
        rQOik.push_back(cv::Point2d(        1583.16,         359.778));
        rQOik.push_back(cv::Point2d(        1583.16,         503.667));
        rQOik.push_back(cv::Point2d(        1583.16,         647.556));
        rQOik.push_back(cv::Point2d(        1588.08,         791.973));
        rQOik.push_back(cv::Point2d(        1599.02,         940.722));
        rQOik.push_back(cv::Point2d(        1610.35,         1094.78));
        rQOik.push_back(cv::Point2d(        1622.09,         1254.43));
        rQOik.push_back(cv::Point2d(        1634.26,         1419.99));
        rQOik.push_back(cv::Point2d(        1679.11,              72));
        rQOik.push_back(cv::Point2d(        1679.11,         215.889));
        rQOik.push_back(cv::Point2d(        1679.11,         359.778));
        rQOik.push_back(cv::Point2d(        1679.11,         503.667));
        rQOik.push_back(cv::Point2d(        1679.11,         647.556));
        rQOik.push_back(cv::Point2d(        1684.79,         791.973));
        rQOik.push_back(cv::Point2d(        1697.43,         940.722));
        rQOik.push_back(cv::Point2d(        1710.51,         1094.78));
        rQOik.push_back(cv::Point2d(        1724.07,         1254.43));
        rQOik.push_back(cv::Point2d(        1738.13,         1419.99));
        rQOik.push_back(cv::Point2d(        1775.05,              72));
        rQOik.push_back(cv::Point2d(        1775.05,         215.889));
        rQOik.push_back(cv::Point2d(        1775.05,         359.778));
        rQOik.push_back(cv::Point2d(        1775.05,         503.667));
        rQOik.push_back(cv::Point2d(        1775.05,         647.556));
        rQOik.push_back(cv::Point2d(         1781.5,         791.973));
        rQOik.push_back(cv::Point2d(        1795.83,         940.722));
        rQOik.push_back(cv::Point2d(        1810.67,         1094.78));
        rQOik.push_back(cv::Point2d(        1826.05,         1254.43));
        rQOik.push_back(cv::Point2d(           1842,         1419.99));
        rQOik.push_back(cv::Point2d(           1871,              72));
        rQOik.push_back(cv::Point2d(           1871,         215.889));
        rQOik.push_back(cv::Point2d(           1871,         359.778));
        rQOik.push_back(cv::Point2d(           1871,         503.667));
        rQOik.push_back(cv::Point2d(           1871,         647.556));
        rQOik.push_back(cv::Point2d(        1878.21,         791.973));
        rQOik.push_back(cv::Point2d(        1894.24,         940.722));
        rQOik.push_back(cv::Point2d(        1910.83,         1094.78));

        std::vector<uchar> mask;

        WHEN("Calling findFundamentalMat")
        {
            double threshold        = 1; // Epipolar error threshold measured in pixels

            // Call findFundamentalMat
            cv::Mat Fkkm1_cv;
            // TODO: Lab 10

            // Check dimensions
            REQUIRE(Fkkm1_cv.type() == CV_64F);
            REQUIRE(Fkkm1_cv.rows == 3);
            REQUIRE(Fkkm1_cv.cols == 3);

            // Get pixel locations in homogeneous coordinates
            Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1;
            Eigen::Matrix<double, 3, Eigen::Dynamic> pk;
            // TODO: Lab 10

            // Interpret fundamental matrix as an Eigen matrix
            Eigen::Map<Eigen::Matrix3d, Eigen::Unaligned, Eigen::Stride<1, 3>> Fkkm1(Fkkm1_cv.ptr<double>(), 3, 3);

            // Calculate normalised epipolar lines
            Eigen::Matrix<double, 3, Eigen::Dynamic> nlk;
            // TODO: Lab 10

            // Calculate epipolar error
            Eigen::RowVectorXd d;
            // TODO: Lab 10

            THEN("Fundamental matrix has correct properties")
            {
                CAPTURE_EIGEN(Fkkm1);
                REQUIRE(Fkkm1.rows() == 3);
                REQUIRE(Fkkm1.cols() == 3);
                REQUIRE(std::abs(Fkkm1.determinant()) < 1e-8);
                
                // Check Fkkm1(:,1)
                CHECK(Fkkm1(0,0) == doctest::Approx(Fkkm1_cv.at<double>(0,0)));
                CHECK(Fkkm1(1,0) == doctest::Approx(Fkkm1_cv.at<double>(1,0)));
                CHECK(Fkkm1(2,0) == doctest::Approx(Fkkm1_cv.at<double>(2,0)));

                // Check Fkkm1(:,2)
                CHECK(Fkkm1(0,1) == doctest::Approx(Fkkm1_cv.at<double>(0,1)));
                CHECK(Fkkm1(1,1) == doctest::Approx(Fkkm1_cv.at<double>(1,1)));
                CHECK(Fkkm1(2,1) == doctest::Approx(Fkkm1_cv.at<double>(2,1)));

                // Check Fkkm1(:,3)
                CHECK(Fkkm1(0,2) == doctest::Approx(Fkkm1_cv.at<double>(0,2)));
                CHECK(Fkkm1(1,2) == doctest::Approx(Fkkm1_cv.at<double>(1,2)));
                CHECK(Fkkm1(2,2) == doctest::Approx(Fkkm1_cv.at<double>(2,2)));
            }

            THEN("Epipolar error is correct")
            {
                //--------------------------------------------------------------------------------
                // Checks for d 
                //--------------------------------------------------------------------------------
                THEN("d is not empty")
                {
                    REQUIRE(d.size()>0);
                    
                    AND_THEN("d has the right dimensions")
                    {
                        REQUIRE(d.rows()==1);
                        REQUIRE(d.cols()==196);
                        AND_THEN("d is correct")
                        {
                            CHECK(std::abs(std::abs(d(0)) -   1.38555833473e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(1)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(2)) -                   2) < 0.012);
                            CHECK(std::abs(std::abs(d(3)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(4)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(5)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(6)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(7)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(8)) -   4.26325641456e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(9)) -   8.52651282912e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(10)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(11)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(12)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(13)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(14)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(15)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(16)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(17)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(18)) -   9.94759830064e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(19)) -   7.81597009336e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(20)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(21)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(22)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(23)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(24)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(25)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(26)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(27)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(28)) -   1.42108547152e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(29)) -   8.52651282912e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(30)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(31)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(32)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(33)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(34)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(35)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(36)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(37)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(38)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(39)) -   1.42108547152e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(40)) -   1.84741111298e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(41)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(42)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(43)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(44)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(45)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(46)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(47)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(48)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(49)) -   5.68434188608e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(50)) -   2.20268248086e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(51)) -   1.42108547152e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(52)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(53)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(54)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(55)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(56)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(57)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(58)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(59)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(60)) -   1.98951966013e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(61)) -   3.12638803734e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(62)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(63)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(64)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(65)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(66)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(67)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(68)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(69)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(70)) -   2.84217094304e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(71)) -   5.40012479178e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(72)) -    6.8212102633e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(73)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(74)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(75)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(76)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(77)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(78)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(79)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(80)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(81)) -   3.97903932026e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(82)) -   7.38964445191e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(83)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(84)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(85)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(86)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(87)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(88)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(89)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(90)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(91)) -   5.68434188608e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(92)) -   1.53477230924e-12) < 0.012);
                            CHECK(std::abs(std::abs(d(93)) -   9.09494701773e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(94)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(95)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(96)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(97)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(98)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(99)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(100)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(101)) -    6.8212102633e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(102)) -   2.27373675443e-12) < 0.012);
                            CHECK(std::abs(std::abs(d(103)) -   1.70530256582e-12) < 0.012);
                            CHECK(std::abs(std::abs(d(104)) -    6.8212102633e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(105)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(106)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(107)) -    6.8212102633e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(108)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(109)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(110)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(111)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(112)) -   9.09494701773e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(113)) -   4.26325641456e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(114)) -   2.84217094304e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(115)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(116)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(117)) -   5.68434188608e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(118)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(119)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(120)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(121)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(122)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(123)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(124)) -   8.52651282912e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(125)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(126)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(127)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(128)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(129)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(130)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(131)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(132)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(133)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(134)) -   5.68434188608e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(135)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(136)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(137)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(138)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(139)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(140)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(141)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(142)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(143)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(144)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(145)) -   3.48165940522e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(146)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(147)) -   3.97903932026e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(148)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(149)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(150)) -   4.54747350886e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(151)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(152)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(153)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(154)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(155)) -    1.4921397451e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(156)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(157)) -   5.11590769747e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(158)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(159)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(160)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(161)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(162)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(163)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(164)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(165)) -   2.55795384874e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(166)) -   2.48689957516e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(167)) -   2.84217094304e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(168)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(169)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(170)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(171)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(172)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(173)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(174)) -   5.68434188608e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(175)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(176)) -   2.84217094304e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(177)) -   5.25801624462e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(178)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(179)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(180)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(181)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(182)) -   1.13686837722e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(183)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(184)) -   5.68434188608e-14) < 0.012);
                            CHECK(std::abs(std::abs(d(185)) -   2.27373675443e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(186)) -   1.84741111298e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(187)) -   3.62376795238e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(188)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(189)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(190)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(191)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(192)) -                   0) < 0.012);
                            CHECK(std::abs(std::abs(d(193)) -   3.41060513165e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(194)) -   1.70530256582e-13) < 0.012);
                            CHECK(std::abs(std::abs(d(195)) -   3.41060513165e-13) < 0.012);
                        }
                    }
                }
            }

            THEN("mask has the correct dimensions")
            {
                REQUIRE(mask.size() == 196);

                AND_THEN("mask has the correct number of inliers")
                {
                    int nInliers = 0;
                    for (int i = 0; i < 196; ++i)
                    {
                        if(mask[i])
                        {
                            nInliers++;
                        }
                    }
                    REQUIRE(nInliers == 195);
                }
            }
        }
    }
}

