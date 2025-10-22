%% Main function to generate tests
function tests = test_GaussianInfoLogIntegral
tests = functiontests(localfunctions);
end

%% Test Functions

function testMeanWithinInterval(testCase)
mu = 2;
sigma = 1;
p = GaussianInfo.fromSqrtMoment(mu, sigma);
a = 0;
b = 3;
assumeTrue(testCase, a <= mu && mu <= b);
actual = p.logIntegral(a, b);
expected = reallog(normcdf(b, mu, sigma) - normcdf(a, mu, sigma));
assertEqual(testCase, actual, expected, 'AbsTol', 1e-12);
end

function testMeanWithinIntervalGradient(testCase)
mu = 2;
sigma = 1;
p = GaussianInfo.fromSqrtMoment(mu, sigma);
a = 0;
b = 3;
assumeTrue(testCase, a <= mu && mu <= b);
[~, actual] = p.logIntegral(a, b);
% expected = (normpdf(a, mu, sigma) - normpdf(b, mu, sigma))/(normcdf(b, mu, sigma) - normcdf(a, mu, sigma));
addpath('./DERIVESTsuite');
[expected, err] = gradest(@(x) GaussianInfo.fromSqrtMoment(x, sigma).logIntegral(a, b), mu);
expected = expected.';
err = err.';
rmpath('./DERIVESTsuite');
assertEqual(testCase, actual, expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected gradient to agree with numerical solution');
end

function testMeanBelowInterval(testCase)
mu = -1;
sigma = 1;
p = GaussianInfo.fromSqrtMoment(mu, sigma);
a = 0;
b = 3;
assumeTrue(testCase, mu < a);
actual = p.logIntegral(a, b);
expected = reallog(normcdf(b, mu, sigma) - normcdf(a, mu, sigma));
assertEqual(testCase, actual, expected, 'AbsTol', 1e-12);
end

function testMeanBelowIntervalGradient(testCase)
mu = -1;
sigma = 1;
p = GaussianInfo.fromSqrtMoment(mu, sigma);
a = 0;
b = 3;
assumeTrue(testCase, mu < a);
[~, actual] = p.logIntegral(a, b);
% expected = (normpdf(a, mu, sigma) - normpdf(b, mu, sigma))/(normcdf(b, mu, sigma) - normcdf(a, mu, sigma));
addpath('./DERIVESTsuite');
[expected, err] = gradest(@(x) GaussianInfo.fromSqrtMoment(x, sigma).logIntegral(a, b), mu);
expected = expected.';
err = err.';
rmpath('./DERIVESTsuite');
assertEqual(testCase, actual, expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected gradient to agree with numerical solution');
end

function testMeanAboveInterval(testCase)
mu = 4;
sigma = 1;
p = GaussianInfo.fromSqrtMoment(mu, sigma);
a = 0;
b = 3;
assumeTrue(testCase, mu > b);
actual = p.logIntegral(a, b);
expected = reallog(normcdf(b, mu, sigma) - normcdf(a, mu, sigma));
assertEqual(testCase, actual, expected, 'AbsTol', 1e-12);
end

function testMeanAboveIntervalGradient(testCase)
mu = 4;
sigma = 1;
p = GaussianInfo.fromSqrtMoment(mu, sigma);
a = 0;
b = 3;
assumeTrue(testCase, mu > b);
[~, actual] = p.logIntegral(a, b);
% expected = (normpdf(a, mu, sigma) - normpdf(b, mu, sigma))/(normcdf(b, mu, sigma) - normcdf(a, mu, sigma));
addpath('./DERIVESTsuite');
[expected, err] = gradest(@(x) GaussianInfo.fromSqrtMoment(x, sigma).logIntegral(a, b), mu);
expected = expected.';
err = err.';
rmpath('./DERIVESTsuite');
assertEqual(testCase, actual, expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected gradient to agree with numerical solution');
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
