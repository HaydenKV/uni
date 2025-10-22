#include <doctest/doctest.h>

TEST_CASE("Basic")
{
    int x = 1;
    SUBCASE("Test subcase 1")
    {
        x = x + 1;
        REQUIRE(x == 2);
    }
    SUBCASE("Test subcase 2")
    {
        REQUIRE(x == 1);
    }
}