%% Main function to generate tests
function tests = test_GaussianSimulate
    tests = functiontests(localfunctions);
end

%% Test Functions
function testGaussianSimulateZeroSqrtCov(testCase)
mu = [3; 1];
S = zeros(2);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
s = rng;    % save seed
rng(1);
x = p.simulate();
rng(s);     % restore seed
assertEqual(testCase, x, mu, 'Expected sample to match mean');
end

function testGaussianSimulateDiagSqrtCov(testCase)
mu = [3; 1];
S = diag([5, 2]);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
m = 100000;
s = rng;    % save seed
rng(1);
X = p.simulate(m);
rng(s);     % restore seed
verifyEqual(testCase, mean(X, 2), p.mean(), 'Expected sample mean to match mean', 'AbsTol', 0.02);
verifyEqual(testCase, cov(X.'), p.cov(), 'Expected sample covariance to match covariance', 'AbsTol', 0.1);
end

function testGaussianSimulateUTSqrtCov(testCase)
mu = [3; 1];
S = [1, -0.3; 0, 0.1];
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
m = 100000;
s = rng;    % save seed
rng(1);
X = p.simulate(m);
rng(s);     % restore seed
verifyEqual(testCase, mean(X, 2), p.mean(), 'Expected sample mean to match mean', 'AbsTol', 0.01);
verifyEqual(testCase, cov(X.'), p.cov(), 'Expected sample covariance to match covariance', 'AbsTol', 0.01);
end

function testCovarianceOverflow(testCase)
n = 2;
mu = zeros(n, 1);
S = 1e300*eye(n);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
assumeFalse(testCase, all(isfinite(p.cov()), 'all'), 'Assume covariance overflows to inf')
m = 1;
s = rng;    % save seed
rng(1);
actual = p.simulate(m);
rng(s);     % restore seed
assertTrue(testCase, all(isfinite(actual)), 'Expected samples to be finite')
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
