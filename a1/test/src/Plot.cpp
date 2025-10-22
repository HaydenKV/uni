#include <doctest/doctest.h>
#include <cmath>
#include <cassert>

// Local copy of static hsv2rgb from Plot.cpp so it’s linkable in tests.
static void hsv2rgb(const double & h, const double & s, const double & v,
                    double & r, double & g, double & b)
{
    assert(0 <= h && h <=  360.0);
    assert(0 <= s && s <=  1.0);
    assert(0 <= v && v <=  1.0);

    double  c, x, r1 = 0, g1 = 0, b1 = 0, m;
    int hp = int(h / 60.);
    c   = v*s;
    x   = c * (1 - std::abs((hp % 2) - 1));

    switch(hp) {
        case 0: r1 = c; g1 = x; b1 = 0; break;
        case 1: r1 = x; g1 = c; b1 = 0; break;
        case 2: r1 = 0; g1 = c; b1 = x; break;
        case 3: r1 = 0; g1 = x; b1 = c; break;
        case 4: r1 = x; g1 = 0; b1 = c; break;
        case 5: r1 = c; g1 = 0; b1 = x; break;
        case 6: r1 = c; g1 = x; b1 = 0; break; // handle h=360°
    }
    m   = v - c;
    r   = r1 + m;
    g   = g1 + m;
    b   = b1 + m;
}

// -------------------------------------------------------------------------------------------------
// hsv2rgb color conversion — boundaries
// Purpose:
//  • testing that hue boundaries and S=0, V=0 behave as expected.
// -------------------------------------------------------------------------------------------------
SCENARIO("hsv2rgb boundary cases")
{
    GIVEN("Edge cases for HSV values")
    {
        double r, g, b;

        WHEN("Hue is 0 and 360 (same color)")
        {
            hsv2rgb(0.0, 1.0, 1.0, r, g, b);
            CHECK(r == doctest::Approx(1.0));
            CHECK(g == doctest::Approx(0.0));
            CHECK(b == doctest::Approx(0.0));

            hsv2rgb(360.0, 1.0, 1.0, r, g, b);
            CHECK(r == doctest::Approx(1.0));
            CHECK(g == doctest::Approx(0.0));
            CHECK(b == doctest::Approx(0.0));
        }

        WHEN("Saturation is 0")
        {
            hsv2rgb(120.0, 0.0, 0.5, r, g, b);
            CHECK(r == doctest::Approx(0.5));
            CHECK(g == doctest::Approx(0.5));
            CHECK(b == doctest::Approx(0.5));
        }

        WHEN("Value is 0")
        {
            hsv2rgb(240.0, 1.0, 0.0, r, g, b);
            CHECK(r == doctest::Approx(0.0));
            CHECK(g == doctest::Approx(0.0));
            CHECK(b == doctest::Approx(0.0));
        }
    }
}  // base boundary tests preserved :contentReference[oaicite:5]{index=5}

// -------------------------------------------------------------------------------------------------
// hsv2rgb continuity and sector behavior
// Purpose:
//  • testing for continuity near 60° and mid-sector dominance.
// -------------------------------------------------------------------------------------------------
SCENARIO("hsv2rgb continuity and sector behavior")
{
    GIVEN("Saturated colors at constant value")
    {
        double r,g,b, r2,g2,b2;

        WHEN("Hue crosses ~60° boundary")
        {
            hsv2rgb(59.9, 1.0, 1.0, r, g, b);
            hsv2rgb(60.1, 1.0, 1.0, r2, g2, b2);

            CHECK(r2 <= r);            // red drops
            CHECK(g2 >= g);            // green rises
            CHECK(b  == doctest::Approx(0.0).epsilon(1e-12));
            CHECK(b2 == doctest::Approx(0.0).epsilon(1e-12));
        }

        WHEN("Mid-sector hues have expected dominance")
        {
            hsv2rgb(120.0, 1.0, 1.0, r, g, b); // green
            CHECK(g == doctest::Approx(1.0));
            CHECK(r == doctest::Approx(0.0));
            CHECK(b == doctest::Approx(0.0));

            hsv2rgb(300.0, 1.0, 1.0, r, g, b); // magenta
            CHECK(r == doctest::Approx(1.0));
            CHECK(b == doctest::Approx(1.0));
            CHECK(g == doctest::Approx(0.0));
        }
    }
}  // continuity additions merged here :contentReference[oaicite:6]{index=6}
