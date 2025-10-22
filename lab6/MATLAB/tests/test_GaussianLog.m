%% Main function to generate tests
function tests = test_GaussianLog
tests = functiontests(localfunctions);
end

%% Test Functions

function testNominal(testCase)
x = [1; 2; 3];
mu = [2; 4; 6];
S = diag([1, 2, 3]);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
expected = -6.04857506884207;
actual = p.log(x);
assertEqual(testCase, actual, expected, 'AbsTol', 1e-12);
end

function testNominalGradient(testCase)
x = [1; 2; 3];
mu = [2; 4; 6];
S = diag([1, 2, 3]);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
[~, actual] = p.log(x);
addpath('./DERIVESTsuite');
[expected, err] = gradest(@(x) p.log(x), x);
expected = expected.';
err = err.';
rmpath('./DERIVESTsuite');
assertEqual(testCase, actual, expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected gradient to agree with numerical solution');
end

function testNominalHessian(testCase)
x = [1; 2; 3];
mu = [2; 4; 6];
S = diag([1, 2, 3]);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
[~, ~, actual] = p.log(x);
addpath('./DERIVESTsuite');
[expected, err] = hessian(@(x) p.log(x), x);
rmpath('./DERIVESTsuite');
assertEqual(testCase, actual, expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected Hessian to agree with numerical solution');
end

function testNominalMultipleX(testCase)
x = [1; 2; 3].*(1:5);
mu = [2; 4; 6];
S = diag([1, 2, 3]);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
expected = [-6.04857506884207, -4.54857506884207, -6.04857506884207, -10.5485750688421, -18.0485750688421];
actual = p.log(x);
assertEqual(testCase, actual, expected, 'AbsTol', 1e-12);
end

function testNegativeS(testCase)
x = [1; 2; 3];
mu = [2; 4; 6];
S = diag([-1, -2, -3]);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
actual = p.log(x);
assertTrue(testCase, isreal(actual), 'Expected log to be real')
end

function testUnderflow(testCase)
x = 0;
mu = sqrt(350*log(10)/pi);  % Approx 16
S = 1/sqrt(2*pi);           % Approx 0.4
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
z = S.'\(x - mu);
l = -0.5*(z.'*z);           % -805.904782547916
assumeEqual(testCase, exp(l), 0, 'Assume exponential underflows to zero');
p = Gaussian.fromSqrtMoment(mu, S);
expected = l; 
actual = p.log(x);
assertEqual(testCase, actual, expected, 'AbsTol', 1e-10);
end

function testDetUnderflow(testCase)
a = 1e-4;  	% Magnitude of st.dev.
n = 100;    % Dimension
S = a*eye(n);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
assumeEqual(testCase, det(S), 0, 'Assume det(S) underflows to zero');
x = zeros(n, 1);
mu = zeros(n, 1);
p = Gaussian.fromSqrtMoment(mu, S);
expected = -n*reallog(a) - n/2*reallog(2*pi);
actual = p.log(x);
assertEqual(testCase, actual, expected, 'AbsTol', 1e-5);
end

function testDetOverflow(testCase)
a = 1e4;    % Magnitude of st.dev.
n = 100;    % Dimension
S = a*eye(n);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
assumeEqual(testCase, det(S), inf, 'Assume det(S) overflows to inf');
x = zeros(n, 1);
mu = zeros(n, 1);
p = Gaussian.fromSqrtMoment(mu, S);
expected = -n*reallog(a) - n/2*reallog(2*pi);
actual = p.log(x);
assertEqual(testCase, actual, expected, 'AbsTol', 1e-5);
end

function testCovarianceOverflow(testCase)
n = 2;
mu = zeros(n, 1);
S = 1e300*eye(n);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
assumeFalse(testCase, all(isfinite(p.cov()), 'all'), 'Assume covariance overflows to inf')
x = zeros(n, 1);
actual = p.log(x);
assertTrue(testCase, isfinite(actual), 'Expected log to be finite')
end

function testCovarianceInverse(testCase)
e = 1.5*sqrt(eps);
mu = [0; e];
S = [1, realsqrt(1 - e^2); 0, e];
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
x = [0; 0];

% V = (x - mu).'/P*(x - mu) = 1 analytically, but not numerically if P is poorly conditioned
ws = warning('off', 'MATLAB:nearlySingularMatrix'); % Disable warning and save warning state
V = (x - mu).'/p.cov()*(x - mu);
warning(ws); % Restore previous warnings
assumeNotEqual(testCase, V, 1.0, 'Assume covariance is poorly conditioned');

n = p.dim();
expected = -0.5 - reallog(e) - n/2*reallog(2*pi);
actual = p.log(x);
assertEqual(testCase, actual, expected, 'AbsTol', 1e-5);
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
