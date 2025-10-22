#include <doctest/doctest.h>
#include <filesystem>
#include <cmath>
#include "../../src/DJIVideoCaption.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

SCENARIO("DJIVideoCaption: Check first caption")
{
    GIVEN("Subtitle file")
    {
        std::filesystem::path captionPath = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("DJI_0121.SRT");
        REQUIRE(std::filesystem::exists(captionPath));

        WHEN("Calling getVideoCaptions")
        {
            std::vector<DJIVideoCaption> djiVideoCaption = getVideoCaptions(captionPath);

            THEN("The correct number of caption frames are found")
            {
                REQUIRE(djiVideoCaption.size() == 6432);

                AND_THEN("frameNum is correct")
                {
                    REQUIRE(djiVideoCaption[0].frameNum == 1);
                    REQUIRE(djiVideoCaption[1].frameNum == 2);
                    REQUIRE(djiVideoCaption[6432-1].frameNum == 6432);
                }

                AND_THEN("time is correct")
                {
                    REQUIRE(djiVideoCaption[0].time == doctest::Approx(0));
                    REQUIRE(djiVideoCaption[1].time == doctest::Approx(0.016));
                    REQUIRE(djiVideoCaption[6432-1].time == doctest::Approx(107.29));
                }

                AND_THEN("iso is correct")
                {
                    REQUIRE(djiVideoCaption[0].iso == 100);
                    REQUIRE(djiVideoCaption[1].iso == 100);
                    REQUIRE(djiVideoCaption[6432-1].iso == 100);
                }

                AND_THEN("shutterHz is correct")
                {
                    REQUIRE(djiVideoCaption[0].shutterHz == doctest::Approx(120));
                    REQUIRE(djiVideoCaption[1].shutterHz == doctest::Approx(120));
                    REQUIRE(djiVideoCaption[6432-1].shutterHz == doctest::Approx(320));
                }

                AND_THEN("fnum is correct")
                {
                    REQUIRE(djiVideoCaption[0].fnum == doctest::Approx(2.80));
                    REQUIRE(djiVideoCaption[1].fnum == doctest::Approx(2.80));
                    REQUIRE(djiVideoCaption[6432-1].fnum == doctest::Approx(2.80));
                }

                AND_THEN("latitude is correct")
                {
                    REQUIRE(djiVideoCaption[0].latitude == doctest::Approx( -32.869698));
                    REQUIRE(djiVideoCaption[1].latitude == doctest::Approx( -32.869698));
                    REQUIRE(djiVideoCaption[6432-1].latitude == doctest::Approx( -32.866047));
                }

                AND_THEN("longitude is correct")
                {
                    REQUIRE(djiVideoCaption[0].longitude == doctest::Approx( 151.688473));
                    REQUIRE(djiVideoCaption[1].longitude == doctest::Approx( 151.688473));
                    REQUIRE(djiVideoCaption[6432-1].longitude == doctest::Approx( 151.683938));
                }

                AND_THEN("altitude is correct")
                {
                    REQUIRE(djiVideoCaption[0].altitude == doctest::Approx(129.789993));
                    REQUIRE(djiVideoCaption[1].altitude == doctest::Approx(129.789993));
                    REQUIRE(djiVideoCaption[6432-1].altitude == doctest::Approx(133.757004));
                }
            }
        }
    }
}
