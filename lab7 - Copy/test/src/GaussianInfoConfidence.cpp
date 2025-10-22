#include <doctest/doctest.h>
#include <Eigen/Core>

#include "../../src/GaussianInfo.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("Inverse cdf of chi^2 distribution")
{
    GIVEN("c = 0.5, nu = 1")
    {
        double c    = 0.5;
        double nu   = 1;

        WHEN("chi2inv(c, nu) is called")
        {
            double f = GaussianInfo<double>::chi2inv(c, nu);

            THEN("Result is the same as MATLAB")
            {
                CHECK(f == doctest::Approx(0.454936423119573));    
            }
        }
    }

    GIVEN("c = 0.5, nu = 2")
    {
        double c    = 0.5;
        double nu   = 2;

        WHEN("chi2inv(c, nu) is called")
        {
            double f = GaussianInfo<double>::chi2inv(c, nu);

            THEN("Result is the same as MATLAB")
            {
                CHECK(f == doctest::Approx(1.386294361119890));    
            }
        }
    }

    GIVEN("c = 0.5, nu = 3")
    {
        double c    = 0.5;
        double nu   = 3;

        WHEN("chi2inv(c, nu) is called")
        {
            double f = GaussianInfo<double>::chi2inv(c, nu);

            THEN("Result is the same as MATLAB")
            {
                CHECK(f == doctest::Approx(2.365973884375338));    
            }
        }
    }

    GIVEN("c = 0.75, nu = 3")
    {
        double c    = 0.75;
        double nu   = 3;
        
        WHEN("chi2inv(c, nu) is called")
        {
            double f = GaussianInfo<double>::chi2inv(c, nu);

            THEN("Result is the same as MATLAB")
            {
                CHECK(f == doctest::Approx(4.108344935632316));    
            }
        }
    }
}

SCENARIO("cdf of standard normal distribution")
{
    GIVEN("x = 0")
    {
        double x = 0;

        WHEN("chi2inv(x) is called")
        {
            double f = GaussianInfo<double>::normcdf(x);

            THEN("Result is the same as MATLAB")
            {
                CHECK(f == doctest::Approx(0.5));    
            }
        }
    }

    GIVEN("x = 1")
    {
        double x = 1;

        WHEN("chi2inv(x) is called")
        {
            double f = GaussianInfo<double>::normcdf(x);

            THEN("Result is the same as MATLAB")
            {
                CHECK(f == doctest::Approx(0.841344746068543));    
            }
        }
    }

    GIVEN("x = 2")
    {
        double x = 2;
        
        WHEN("chi2inv(x) is called")
        {
            double f = GaussianInfo<double>::normcdf(x);

            THEN("Result is the same as MATLAB")
            {
                CHECK(f == doctest::Approx(0.977249868051821));    
            }
        }
    }
}

SCENARIO("GaussianInfo confidence ellipse")
{
    GIVEN("Gaussian p, where S is identity, mu is [0.5; 1.5]")
    {
        Eigen::VectorXd mu(2);
        Eigen::MatrixXd S(2, 2);

        // mu - [3 x 1]: 
        mu <<               0.5,
                            1.5;

        // S - [3 x 3]: 
        S <<                  1,                 0,
                              0,                 1;
        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        REQUIRE(p.dim() == 2);

        WHEN("X = p.confidenceEllipse(3, 100) is called")
        {
            double nSigma = 3.0;
            int nSamples = 100;
            Eigen::Matrix<double, 2, Eigen::Dynamic> X = p.confidenceEllipse(nSigma, nSamples);

            THEN("X has nSamples columns")
            {
                REQUIRE(X.cols() == nSamples);

                AND_THEN("All points lie on the confidence region boundary")
                {
                    CAPTURE_EIGEN(X);
                    Eigen::Vector2d mean = X(Eigen::all, Eigen::seqN(0, nSamples - 1)).rowwise().sum()/(nSamples - 1.0); // Remove either start or end point, since they overlap
                    CAPTURE_EIGEN(mean);
                    CHECK( ((mean - mu).array().abs() < 1e-10).all() );
                }

                AND_THEN("All points lie between the 2.9 and 3.1 sigma confidence regions")
                {
                    for (Eigen::Index i = 0; i < X.cols(); ++i)
                    {
                        CAPTURE(i);
                        CAPTURE_EIGEN(X.col(i));
                        CHECK_FALSE(p.isWithinConfidenceRegion(X.col(i), 2.9));
                        CHECK(p.isWithinConfidenceRegion(X.col(i), 3.1));
                    }
                }
            }
        }
    }

    GIVEN("Gaussian p, where S is [1, 0.3; 0, 0.1], mu is zero")
    {
        Eigen::VectorXd mu(2);
        Eigen::MatrixXd S(2, 2);

        // mu - [3 x 1]: 
        mu <<                 0,
                              0;

        // S - [3 x 3]: 
        S <<                  1,               0.3,
                              0,               0.1;
        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        REQUIRE(p.dim() == 2);

        WHEN("X = p.confidenceEllipse(3, 100) is called")
        {
            double nSigma = 3.0;
            int nSamples = 100;
            Eigen::Matrix<double, 2, Eigen::Dynamic> X = p.confidenceEllipse(nSigma, nSamples);

            THEN("X has nSamples columns")
            {
                REQUIRE(X.cols() == nSamples);

                AND_THEN("All points lie on the confidence region boundary")
                {
                    Eigen::Matrix<double, 2, Eigen::Dynamic> W = S.triangularView<Eigen::Upper>().transpose().solve(X);
                    Eigen::RowVectorXd r = W.colwise().norm();
                    
                    CAPTURE_EIGEN(r);
                    CHECK( ((r.array() - 3.43935431177144).abs() < 1e-10).all() );
                }

                AND_THEN("All points lie between the 2.9 and 3.1 sigma confidence regions")
                {
                    for (Eigen::Index i = 0; i < X.cols(); ++i)
                    {
                        CAPTURE(i);
                        CAPTURE_EIGEN(X.col(i));
                        CHECK_FALSE(p.isWithinConfidenceRegion(X.col(i), 2.9));
                        CHECK(p.isWithinConfidenceRegion(X.col(i), 3.1));
                    }
                }
            }
        }
    }
}

SCENARIO("GaussianInfo quadric surface coefficients")
{
    GIVEN("Gaussian p, where S is identity, mu is zero")
    {
        Eigen::VectorXd mu(3);
        Eigen::MatrixXd S(3, 3);

        // mu - [3 x 1]: 
        mu <<                  0,
                              -0,
                               0;

        // S - [3 x 3]: 
        S <<                  1,                 0,                 0,
                              0,                 1,                 0,
                              0,                 0,                 1;
        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        REQUIRE(p.dim() == 3);

        WHEN("p.quadricSurface(3) is called")
        {
            Eigen::Matrix4d Q = p.quadricSurface(3);
            //--------------------------------------------------------------------------------
            // Checks for Q 
            //--------------------------------------------------------------------------------
            THEN("Q is not empty")
            {
                REQUIRE(Q.size() > 0);
                
                AND_THEN("Q has the right dimensions")
                {
                    REQUIRE(Q.rows() == 4);
                    REQUIRE(Q.cols() == 4);

                    AND_THEN("Q matches expected values")
                    {
                        Eigen::Matrix4d Q_exp;
                        Q_exp << 1, 0, 0, 0,
                                 0, 1, 0, 0,
                                 0, 0, 1, 0,
                                 0, 0, 0, -14.1564136091267;
                        CAPTURE_EIGEN(Q);
                        CAPTURE_EIGEN(Q_exp);
                        CHECK(Q.isApprox(Q_exp));
                    }
                }
            }
        }
    }

    GIVEN("Gaussian p, where S = diag(4, 5, 6), mu = [1; 2; 3]")
    {
        Eigen::MatrixXd S(3, 3);
        Eigen::VectorXd mu(3);

        // mu - [3 x 1]: 
        mu <<                  1,
                               2,
                               3;

        // S - [3 x 3]: 
        S <<                  4,                 0,                 0,
                              0,                 5,                 0,
                              0,                 0,                 6;
        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);
        REQUIRE(p.dim() == 3);

        WHEN("p.quadricSurface(3) is called")
        {
            Eigen::Matrix4d Q = p.quadricSurface(3);
            //--------------------------------------------------------------------------------
            // Checks for Q 
            //--------------------------------------------------------------------------------
            THEN("Q is not empty")
            {
                REQUIRE(Q.size() > 0);
                
                AND_THEN("Q has the right dimensions")
                {
                    REQUIRE(Q.rows() == 4);
                    REQUIRE(Q.cols() == 4);

                    AND_THEN("Q matches expected values")
                    {
                        Eigen::Matrix4d Q_exp;
                        Q_exp << 1.0/16.0, 0, 0, -1.0/16.0,
                                 0, 1.0/25.0, 0, -2.0/25.0,
                                 0, 0, 1.0/36.0, -1.0/12.0,
                                 -1.0/16.0, -2.0/25.0, -1.0/12.0, -13.6839136091267;
                        CAPTURE_EIGEN(Q);
                        CAPTURE_EIGEN(Q_exp);
                        CHECK(Q.isApprox(Q_exp));
                    }
                }
            }
        }
    }

    GIVEN("Gaussian p, where S is upper triangular, mu = [1; 2; 3]")
    {
        Eigen::VectorXd mu(3);
        Eigen::MatrixXd S(3, 3);

        // mu - [3 x 1]: 
        mu <<                  1,
                               2,
                               3;

        // S - [3 x 3]: 
        S <<      -0.6490137652,      -1.109613039,     -0.5586807645,
                              0,       -0.84555124,      0.1783802258,
                              0,                 0,     -0.1968614465;
        auto p = GaussianInfo<double>::fromSqrtMoment(mu, S);

        WHEN("p.quadricSurface(3) is called")
        {
            Eigen::Matrix4d Q = p.quadricSurface(3);
            //--------------------------------------------------------------------------------
            // Checks for Q 
            //--------------------------------------------------------------------------------
            THEN("Q is not empty")
            {
                REQUIRE(Q.size() > 0);
                
                AND_THEN("Q has the right dimensions")
                {
                    REQUIRE(Q.rows() == 4);
                    REQUIRE(Q.cols() == 4);

                    AND_THEN("Q matches expected values")
                    {
                        Eigen::Matrix4d Q_exp;
                        Q_exp <<  44.9627201133221,  -9.0406493633664, -31.5188988229375,  67.6752750822232,
                                  -9.0406493633664,  2.54708314580204,  5.44359034070936, -12.3842879503658,
                                 -31.5188988229375,  5.44359034070936,   25.803502277206, -56.7787886900993,
                                  67.6752750822232, -12.3842879503658, -56.7787886900993,   113.27325327968;
                        CAPTURE_EIGEN(Q);
                        CAPTURE_EIGEN(Q_exp);
                        CHECK(Q.isApprox(Q_exp));
                    }
                }
            }
        }
    }
}