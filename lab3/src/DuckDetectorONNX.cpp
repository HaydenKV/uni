#include <cstdint>
#include <string>
#include <print>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "DuckDetectorONNX.h"

const std::size_t DuckDetectorONNX::num_inputs = 1;
const std::size_t DuckDetectorONNX::num_outputs = 2;
const char * DuckDetectorONNX::input_names[] = {"input"};
const char * DuckDetectorONNX::output_names[] = {"class_scores", "mask_probs"};

DuckDetectorONNX::DuckDetectorONNX(const std::string & model_path)
    : env(ORT_LOGGING_LEVEL_ERROR, "test"), // ORT_LOGGING_LEVEL_WARNING or ORT_LOGGING_LEVEL_ERROR
      session_options(),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      input_shape{1, 3, 512, 512}
{
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // ONNX Runtime version and provider information
    std::println("ONNX Runtime version: {}", Ort::GetVersionString());
    
    // List available execution providers
    std::println("Available ONNX Runtime providers:");
    auto providers = Ort::GetAvailableProviders();
    for (const auto& provider : providers)
    {
        std::println("  - {}", provider);
    }

    if (std::ranges::contains(providers, "CoreMLExecutionProvider"))
    {
        // Try to append CoreML provider using the new API
        try
        {
            std::unordered_map<std::string, std::string> provider_options;
            provider_options["ModelFormat"] = "MLProgram";  // Use MLProgram format for better performance
            provider_options["MLComputeUnits"] = "ALL";     // Enable all compute units (CPU, GPU, ANE)
            provider_options["RequireStaticInputShapes"] = "0";

            session_options.AppendExecutionProvider("CoreML", provider_options);
            std::println("CoreML provider enabled successfully");
        }
        catch (const std::exception& e)
        {
            std::println("Failed to enable CoreML provider: {}", e.what());
        }
    }

    if (std::ranges::contains(providers, "CUDAExecutionProvider"))
    {
        // Try to append CUDA provider using V2 API with error handling
        try
        {
            OrtCUDAProviderOptionsV2* cuda_options = nullptr;
            const auto& api = Ort::GetApi();
            
            Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));
            
            std::vector<const char*> keys{"device_id", "arena_extend_strategy"};
            std::vector<const char*> values{"0", "kNextPowerOfTwo"};
            Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size()));

            Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(session_options), cuda_options));
            
            // Clean up
            api.ReleaseCUDAProviderOptions(cuda_options);
            
            std::println("CUDA provider enabled successfully");
        }
        catch (const Ort::Exception& e)
        {
            std::println("Failed to enable CUDA provider: {}", e.what());
        }
    }

    // Add TensorRT RTX provider support
    if (std::ranges::contains(providers, "NvTensorRTRTXExecutionProvider"))
    {
        try
        {
            const auto& api = Ort::GetApi();
            
            std::vector<const char*> option_keys = {
                "device_id",
                "nv_cuda_graph_enable"
            };
            std::vector<const char*> option_values = {
                "0",    // Use GPU 0
                "1"     // Enable CUDA graph for better performance
            };

            Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider(
                static_cast<OrtSessionOptions*>(session_options), 
                "NvTensorRtRtx", 
                option_keys.data(), 
                option_values.data(), 
                option_keys.size()
            ));
            
            std::println("TensorRT RTX provider enabled successfully");
        }
        catch (const Ort::Exception& e)
        {
            std::println("Failed to enable TensorRT RTX provider: {}", e.what());
        }
    }

    // Initialize session after configuring providers
    session = Ort::Session(env, model_path.c_str(), session_options);
}

cv::Mat DuckDetectorONNX::detect(const cv::Mat & image)
{
    cv::Mat imgout;
    image.convertTo(imgout, CV_8UC3, 0.5, 0); // Darken the image

    std::vector<float> input_tensor_values(3 * 512 * 512);
    preprocess(image, input_tensor_values);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, num_inputs, output_names, num_outputs);

    auto & class_scores = output_tensors[0];
    auto & mask_probs = output_tensors[1];

    auto class_scores_shape = class_scores.GetTensorTypeAndShapeInfo().GetShape();
    auto mask_probs_shape = mask_probs.GetTensorTypeAndShapeInfo().GetShape();

    float * class_scores_data = class_scores.GetTensorMutableData<float>();
    float * mask_probs_data = mask_probs.GetTensorMutableData<float>();

    std::vector<float> class_scores_vec(class_scores_data, class_scores_data + class_scores.GetTensorTypeAndShapeInfo().GetElementCount());
    std::vector<float> mask_probs_vec(mask_probs_data, mask_probs_data + mask_probs.GetTensorTypeAndShapeInfo().GetElementCount());

    postprocess(class_scores_vec, mask_probs_vec, class_scores_shape, mask_probs_shape, imgout);

    return imgout;
}
