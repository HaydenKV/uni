#ifndef DUCKDETECTORONNX_H
#define DUCKDETECTORONNX_H

#include <cstdint>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include "DuckDetectorBase.h"

class DuckDetectorONNX : public DuckDetectorBase
{
public:
    explicit DuckDetectorONNX(const std::string & model_path);
    virtual cv::Mat detect(const cv::Mat & image) override;

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info;

    static const char * input_names[];
    static const char * output_names[];
    static const std::size_t num_inputs;
    static const std::size_t num_outputs;
    std::vector<std::int64_t> input_shape;
};

#endif
