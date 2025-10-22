%% Main function to generate tests
function tests = test_GaussianFromSamples
    tests = functiontests(localfunctions);
end

%% Test Functions
function testGaussianFromSamplesMean(testCase)
    X = [ -0.15508     -1.0443      -1.1714       0.92622   -0.55806;
           0.61212     -0.34563     -0.68559     -1.4817    -0.028453];
    p = Gaussian.fromSamples(X);
    mu_actual = p.mean();
    mu_expected = [-0.40052; -0.38585];
    verifyEqual(testCase, mu_actual, mu_expected, 'AbsTol', 1e-5, 'Expected sample mean to match oracle');
end

function testGaussianFromSamplesCov(testCase)
    X = [ -0.15508     -1.0443      -1.1714       0.92622   -0.55806;
           0.61212     -0.34563     -0.68559     -1.4817    -0.028453];
    p = Gaussian.fromSamples(X);
    P_actual = p.cov();
    P_expected = [ 0.7135     -0.26502;
                  -0.26502     0.60401];
    verifyEqual(testCase, P_actual, P_expected, 'AbsTol', 1e-5, 'Expected sample covariance to match oracle');
end

%% Optional file fixtures  
function setupOnce(testCase)  % do not change function name
    addpath ../
end

function teardownOnce(testCase)  % do not change function name
    rmpath ../
end
%% Optional fresh fixtures  
function setup(testCase)  % do not change function name
end

function teardown(testCase)  % do not change function name
end