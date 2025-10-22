%% Main function to generate tests
function tests = test_GaussianMarginal
tests = functiontests(localfunctions);
end

%% Test Functions

function testMarginalHead(testCase)
mu = [1; 2; 3; 4; 5];
S = [ ...
    10, 11, 12, 13, 14; ...
     0, 15, 16, 17, 18; ...
     0,  0, 19, 20, 21; ...
     0,  0,  0, 23, 24; ...
     0,  0,  0,  0, 25];
p = Gaussian.fromSqrtMoment(mu, S);
idx = [1, 2];
runMarginalCase(testCase, p, idx);
end

function testMarginalTail(testCase)
mu = [1; 2; 3; 4; 5];
S = [ ...
    10, 11, 12, 13, 14; ...
     0, 15, 16, 17, 18; ...
     0,  0, 19, 20, 21; ...
     0,  0,  0, 23, 24; ...
     0,  0,  0,  0, 25];
p = Gaussian.fromSqrtMoment(mu, S);
idx = [4, 5];
runMarginalCase(testCase, p, idx);
end

function testMarginalSegment(testCase)
mu = [1; 2; 3; 4; 5];
S = [ ...
    10, 11, 12, 13, 14; ...
     0, 15, 16, 17, 18; ...
     0,  0, 19, 20, 21; ...
     0,  0,  0, 23, 24; ...
     0,  0,  0,  0, 25];
p = Gaussian.fromSqrtMoment(mu, S);
idx = [2, 3, 4];
runMarginalCase(testCase, p, idx);
end

function testMarginalNoncontiguous(testCase)
mu = [1; 2; 3; 4; 5];
S = [ ...
    10, 11, 12, 13, 14; ...
     0, 15, 16, 17, 18; ...
     0,  0, 19, 20, 21; ...
     0,  0,  0, 23, 24; ...
     0,  0,  0,  0, 25];
p = Gaussian.fromSqrtMoment(mu, S);
idx = [1, 3, 5];
runMarginalCase(testCase, p, idx);
end

function testMarginalNoncontiguousNonascending(testCase)
mu = [1; 2; 3; 4; 5];
S = [ ...
    10, 11, 12, 13, 14; ...
     0, 15, 16, 17, 18; ...
     0,  0, 19, 20, 21; ...
     0,  0,  0, 23, 24; ...
     0,  0,  0,  0, 25];
p = Gaussian.fromSqrtMoment(mu, S);
idx = [5, 3, 1];
runMarginalCase(testCase, p, idx);
end

function testMarginalCovarianceOverflow(testCase)
n = 2;
mu = zeros(n, 1);
S = 1e300*eye(n);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
assumeFalse(testCase, all(isfinite(p.cov()), 'all'), 'Assume covariance overflows to inf')
idx = [1, 2];
pm = p.marginal(idx);
assertTrue(testCase, all(isfinite(pm.sqrtCov()), 'all'), 'Expected marginal square-root covariance to be finite')
end

function runMarginalCase(testCase, p, idx)
mu = p.mean();
S = p.sqrtCov();
assumeTrue(testCase, istriu(S), 'Expected S to be upper triangular');

pm = p.marginal(idx);
mum_actual = pm.mean();
Sm_actual = pm.sqrtCov();

mum_expected = mu(idx);
assertEqual(testCase, mum_actual, mum_expected, 'Expected marginal mean to match expected result', 'AbsTol', 1e-12);

assertTrue(testCase, istriu(Sm_actual), 'Expected marginal sqrt cov to be upper triangular');

P = p.cov();            % S.'*S
Pm_actual = pm.cov();   % Sm_actual.'*Sm_actual
Pm_expected = P(idx, idx);
assertEqual(testCase, Pm_actual, Pm_expected, 'Expected marginal cov to match expected result', 'AbsTol', 1e-10);
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