#include <doctest/doctest.h>
#include <algorithm>
#include <vector>
#include <string>

#ifdef WITH_ONNX
#include <onnxruntime_cxx_api.h>
SCENARIO("ONNX Runtime providers")
{
    GIVEN("ONNX Runtime is available")
    {
        WHEN("getting available execution providers")
        {
            auto providers = Ort::GetAvailableProviders();
            
            THEN("providers list should not be empty")
            {
                REQUIRE_FALSE(providers.empty());

                AND_THEN("CPUExecutionProvider should be available")
                {
                    auto it = std::find(providers.begin(), providers.end(), "CPUExecutionProvider");
                    REQUIRE(it != providers.end());
                }
            }
        }
    }
}
#else
SCENARIO("ONNX Runtime providers" * doctest::skip())
{
    // Test case is skipped when ONNX Runtime is not available
}
#endif