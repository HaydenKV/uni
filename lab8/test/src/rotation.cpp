#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#include "../../src/rotation.hpp"

SCENARIO("rotx")
{
    GIVEN("x = 0")
    {
        Eigen::Matrix3d R;
        double x = 0;

        WHEN("Calling R = rotx(x)")
        {
            R = rotx(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                  -0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = rotx(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotx(x, dRdx);

            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                  -0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                   0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(                  -0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   1));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                  -1));
                        CHECK(dRdx(2,2) == doctest::Approx(                  -0));
                    }
                }
            }        
        }
    }

    GIVEN("x = 0.1")
    {
        Eigen::Matrix3d R;
        double x = 0.1;

        WHEN("Calling R = rotx(x)")
        {
            R = rotx(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(      0.995004165278));
                        CHECK(R(2,1) == doctest::Approx(     0.0998334166468));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(    -0.0998334166468));
                        CHECK(R(2,2) == doctest::Approx(      0.995004165278));
                    }
                }
            }
        }

        WHEN("Calling R = rotx(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotx(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);
                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(      0.995004165278));
                        CHECK(R(2,1) == doctest::Approx(     0.0998334166468));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(    -0.0998334166468));
                        CHECK(R(2,2) == doctest::Approx(      0.995004165278));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                   0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(    -0.0998334166468));
                        CHECK(dRdx(2,1) == doctest::Approx(      0.995004165278));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(     -0.995004165278));
                        CHECK(dRdx(2,2) == doctest::Approx(    -0.0998334166468));
                    }
                }
            }
        }
    }

    GIVEN("x = pi*5/3")
    {
        Eigen::Matrix3d R;
        double x = M_PI*5/3;

        WHEN("Calling R = rotx(x)")
        {
            R = rotx(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                 0.5));
                        CHECK(R(2,1) == doctest::Approx(     -0.866025403784));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(      0.866025403784));
                        CHECK(R(2,2) == doctest::Approx(                 0.5));
                    }
                }
            }
        }

        WHEN("Calling R = rotx(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotx(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                 0.5));
                        CHECK(R(2,1) == doctest::Approx(     -0.866025403784));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(      0.866025403784));
                        CHECK(R(2,2) == doctest::Approx(                 0.5));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                   0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(      0.866025403784));
                        CHECK(dRdx(2,1) == doctest::Approx(                 0.5));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                -0.5));
                        CHECK(dRdx(2,2) == doctest::Approx(      0.866025403784));
                    }
                }
            }
        }
    }
}


SCENARIO("roty")
{
    GIVEN("x = 0")
    {
        Eigen::Matrix3d R;
        double x = 0;

        WHEN("Calling R = roty(x)")
        {
            R = roty(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                  -0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = roty(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = roty(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                  -0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                  -0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                  -1));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(                   0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   1));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(                  -0));
                    }
                }
            }
        }
    }

    GIVEN("x = 0.1")
    {

        Eigen::Matrix3d R;
        double x = 0.1;

        WHEN("Calling R = roty(x)")
        {
            R = roty(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.995004165278));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(    -0.0998334166468));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     0.0998334166468));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(      0.995004165278));
                    }
                }
            }
        }

        WHEN("Calling R = roty(x, dRdx)"){
            Eigen::Matrix3d dRdx;
            R = roty(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.995004165278));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(    -0.0998334166468));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     0.0998334166468));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(      0.995004165278));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty"){
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions"){
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);
                    AND_THEN("dRdx is correct"){

                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(    -0.0998334166468));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(     -0.995004165278));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(                   0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(      0.995004165278));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(    -0.0998334166468));

                    }
                }
            }        
        }
    }

    GIVEN("x = pi*5/3")
    {
        Eigen::Matrix3d R;
        double x = M_PI*5/3;

        WHEN("Calling R = roty(x)")
        {
            R = roty(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                 0.5));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(      0.866025403784));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     -0.866025403784));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                 0.5));
                    }
                }
            }
        }

        WHEN("Calling R = roty(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = roty(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                 0.5));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(      0.866025403784));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     -0.866025403784));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                 0.5));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(      0.866025403784));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                -0.5));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(                   0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                 0.5));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(      0.866025403784));

                    }
                }
            }
        }
    }
}

SCENARIO("rotz")
{
    GIVEN("x = 0")
    {
        Eigen::Matrix3d R;
        double x = 0;

        WHEN("Calling R = rotz(x)")
        {
            R = rotz(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = rotz(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotz(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                  -0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                  -0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   1));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                  -1));
                        CHECK(dRdx(1,1) == doctest::Approx(                  -0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("x = 0.1")
    {
        Eigen::Matrix3d R;
        double x = 0.1;

        WHEN("Calling R = rotz(x)")
        {
            R = rotz(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions"){

                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.995004165278));
                        CHECK(R(1,0) == doctest::Approx(     0.0998334166468));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(    -0.0998334166468));
                        CHECK(R(1,1) == doctest::Approx(      0.995004165278));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = rotz(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotz(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.995004165278));
                        CHECK(R(1,0) == doctest::Approx(     0.0998334166468));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(    -0.0998334166468));
                        CHECK(R(1,1) == doctest::Approx(      0.995004165278));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(    -0.0998334166468));
                        CHECK(dRdx(1,0) == doctest::Approx(      0.995004165278));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(     -0.995004165278));
                        CHECK(dRdx(1,1) == doctest::Approx(    -0.0998334166468));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("x = pi*5/3")
    {
        Eigen::Matrix3d R;
        double x = M_PI*5/3;

        WHEN("Calling R = rotz(x)")
        {
            R = rotz(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                 0.5));
                        CHECK(R(1,0) == doctest::Approx(     -0.866025403784));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(      0.866025403784));
                        CHECK(R(1,1) == doctest::Approx(                 0.5));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = rotz(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotz(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                 0.5));
                        CHECK(R(1,0) == doctest::Approx(     -0.866025403784));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(      0.866025403784));
                        CHECK(R(1,1) == doctest::Approx(                 0.5));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(      0.866025403784));
                        CHECK(dRdx(1,0) == doctest::Approx(                 0.5));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                -0.5));
                        CHECK(dRdx(1,1) == doctest::Approx(      0.866025403784));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }
}

SCENARIO("rpy2rot")
{
    GIVEN("Theta = [1; 2; 3]")
    {
        Eigen::Matrix3d R;
        Eigen::Vector3d Theta;
        // Theta - [3 x 1]: 
        Theta <<                    1,
                                    2,
                                    3;

        WHEN("Calling R = rpy2rot(Theta)")
        {
            R = rpy2rot(Theta);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.411982245666));
                        CHECK(R(1,0) == doctest::Approx(    -0.0587266449276));
                        CHECK(R(2,0) == doctest::Approx(     -0.909297426826));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(     -0.833737651774));
                        CHECK(R(1,1) == doctest::Approx(     -0.426917621276));
                        CHECK(R(2,1) == doctest::Approx(     -0.350175488374));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     -0.367630462925));
                        CHECK(R(1,2) == doctest::Approx(      0.902381585483));
                        CHECK(R(2,2) == doctest::Approx(     -0.224845095366));
                    }
                }
            }
        }
    }
}

SCENARIO("rot2rpy")
{
    GIVEN("R is identity")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<                    1,                 0,                 0,
                                0,                 1,                 0,
                                0,                 0,                 1;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                
                AND_THEN("Theta has the right dimensions")
                {
                    REQUIRE(Theta.rows()==3);
                    REQUIRE(Theta.cols()==1);

                    AND_THEN("Theta is correct")
                    {
                        // Theta(:,1)
                        CHECK(Theta(0,0) == doctest::Approx(                   0));
                        CHECK(Theta(1,0) == doctest::Approx(                  -0));
                        CHECK(Theta(2,0) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("R is rotation about x axis by 0.5 rad")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<                    1,                 0,                 0,
                                0,      0.8775825619,     -0.4794255386,
                                0,      0.4794255386,      0.8775825619;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                
                AND_THEN("Theta has the right dimensions")
                {
                    REQUIRE(Theta.rows()==3);
                    REQUIRE(Theta.cols()==1);

                    AND_THEN("Theta is correct")
                    {
                        // Theta(:,1)
                        CHECK(Theta(0,0) == doctest::Approx(                 0.5));
                        CHECK(Theta(1,0) == doctest::Approx(                  -0));
                        CHECK(Theta(2,0) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("R is rotation about y axis by 0.5 rad")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<         0.8775825619,                 0,      0.4794255386,
                                0,                 1,                 0,
                    -0.4794255386,                 0,      0.8775825619;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                
                AND_THEN("Theta has the right dimensions")
                {
                    REQUIRE(Theta.rows()==3);
                    REQUIRE(Theta.cols()==1);

                    AND_THEN("Theta is correct")
                    {
                        // Theta(:,1)
                        CHECK(Theta(0,0) == doctest::Approx(                   0));
                        CHECK(Theta(1,0) == doctest::Approx(                 0.5));
                        CHECK(Theta(2,0) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("R is rotation about z axis by 0.5 rad")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<         0.8775825619,     -0.4794255386,                 0,
                     0.4794255386,      0.8775825619,                 0,
                                0,                 0,                 1;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                
                AND_THEN("Theta has the right dimensions")
                {
                    REQUIRE(Theta.rows()==3);
                    REQUIRE(Theta.cols()==1);

                    AND_THEN("Theta is correct")
                    {
                        // Theta(:,1)
                        CHECK(Theta(0,0) == doctest::Approx(                   0));
                        CHECK(Theta(1,0) == doctest::Approx(                  -0));
                        CHECK(Theta(2,0) == doctest::Approx(                 0.5));
                    }
                }
            }
        }
    }

    GIVEN("R is a general rotation")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<         0.4119822457,     -0.8337376518,     -0.3676304629,
                   -0.05872664493,     -0.4269176213,      0.9023815855,
                    -0.9092974268,     -0.3501754884,     -0.2248450954;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                Eigen::Matrix3d R_act = rpy2rot(Theta);

                AND_THEN("rot2rpy is an inverse mapping")
                {
                    // R_act(:,1)
                    CHECK(R_act(0,0) == doctest::Approx(R(0,0)));
                    CHECK(R_act(1,0) == doctest::Approx(R(1,0)));
                    CHECK(R_act(2,0) == doctest::Approx(R(2,0)));

                    // R_act(:,2)
                    CHECK(R_act(0,1) == doctest::Approx(R(0,1)));
                    CHECK(R_act(1,1) == doctest::Approx(R(1,1)));
                    CHECK(R_act(2,1) == doctest::Approx(R(2,1)));

                    // R_act(:,3)
                    CHECK(R_act(0,2) == doctest::Approx(R(0,2)));
                    CHECK(R_act(1,2) == doctest::Approx(R(1,2)));
                    CHECK(R_act(2,2) == doctest::Approx(R(2,2)));
                }
            }
        }
    }
}
