#include <cstdlib>
#include <cassert>
#include <stdexcept>
#include <print>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    std::println("Hello world!");

    //
    // Example: Compile-time assertion
    //
    // Uncomment the following line to create a build error:
    // static_assert(1 == 2, "A message that is included if the condition is false at compile time");

    //
    // Example: Run-time assertion
    //
    int one = 1, two = one + one;
    // Uncomment the following line to create an assertion failure (debug build only)
    // assert(one == two && "A cheeky hack to include a message in a run-time assertion failure");

    //
    // Example: Unhandled exception
    //
    // Uncomment the following line to create an unhandled exception:
    // throw std::runtime_error("An unhandled exception");

    //
    // Example: out-of-bounds access using a stack array
    //
    double stack_array[10];
    // Uncomment the following line to create a segmentation fault (maybe):
    // stack_array[1000000] = 1.0;

    //
    // Example: out-of-bounds access using a heap array
    //
    double *heap_array = new double[10];
    // Uncomment the following line to create a segmentation fault (maybe):
    // heap_array[1000000] = 1.0;
    delete[] heap_array;

    //
    // Example: out-of-bounds access using a C++ STL container
    //
    std::vector<double> v(10);
    // Uncomment the following line to create a segmentation fault (maybe):
    // v[1000000] = 1.0;
    // Uncomment the following line to throw a std::out_of_range exception (since std::vector::at has bounds checking)
    // v.at(1000000) = 1.0;

    //
    // Example: dereferencing a null pointer
    //
    double *p = nullptr;
    // Uncomment the following line to create a segmentation fault:
    // p[10] = 1.0; // Pro tip: Put this in a kernel mode driver to reproduce the 2024 CrowdStrike incident (https://en.wikipedia.org/wiki/2024_CrowdStrike_incident)

    //
    // Example: out-of-bounds access to an OpenCV matrix
    //
    cv::Mat M(10, 10, CV_64FC1);                // A 10x10 matrix of double-precision (64-bit) floating point numbers
    // Uncomment the following line to create a segmentation fault (maybe):
    // M.at<double>(1000, 1000) = 1.0;

    //
    // Example: Matrix multiplication dimension mismatch using an OpenCV matrix
    //
    cv::Mat P(10, 5, CV_64FC1), Q(7, 3, CV_64FC1);
    cv::Mat R;
    // Uncomment the following line to throw a cv::Exception due to dimension mismatch
    // cv::multiply(P, Q, R);

    //
    // Example: out-of-bounds access to an eigen3 matrix
    //
    Eigen::MatrixXd A(10, 10);                  // A 10x10 matrix of double-precision (64-bit) floating point numbers
    // Uncomment the following line to create an assertion failure in a debug build, maybe a segmentation fault in a release build
    // A(1000, 1000) = 1.0;

    //
    // Example: Matrix multiplication dimension mismatch using eigen3
    //
    Eigen::MatrixXd B(10, 5), C(7, 3);
    // Uncomment the following line to produce an assertion failure due to dimension mismatch
    // Eigen::MatrixXd D = B*C;
    
    //
    // Draw a smiley face and display it with cv::imshow
    //
    cv::Mat smiley(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));  // White background

    // Draw face
    cv::circle(smiley, cv::Point(200, 200), 150, cv::Scalar(0, 255, 255), -1);  // Yellow face
    cv::circle(smiley, cv::Point(200, 200), 150, cv::Scalar(0, 0, 0), 5);  // Black border

    // Draw eyes
    cv::circle(smiley, cv::Point(150, 150), 30, cv::Scalar(0, 0, 0), -1);  // Left eye
    cv::circle(smiley, cv::Point(250, 150), 30, cv::Scalar(0, 0, 0), -1);  // Right eye

    // Draw a friendlier smile
    cv::ellipse(smiley, cv::Point(200, 220), cv::Size(80, 60), 0, 0, 180, cv::Scalar(0, 0, 0), 15);

    // Display the image
    cv::imshow("Smiley Face - press ESC to exit", smiley);
    
    // Wait for ESC key press to exit
    while (cv::waitKey(0) != 27);  // 27 is the ASCII code for ESC
    cv::destroyAllWindows();  // Close the window

    return EXIT_SUCCESS;
}

