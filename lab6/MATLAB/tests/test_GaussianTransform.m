%% Main function to generate tests
function tests = test_GaussianTransform
    tests = functiontests(localfunctions);
end

%% Test Functions
function testGaussianTransformAffine(testCase)
mux = [1; 3];
Sx = [0.1, 0.01; 0, 0.01];
assumeTrue(testCase, istriu(Sx), 'Assume Sx is upper triangular');
px = Gaussian.fromSqrtMoment(mux, Sx);

% Assume Jacobian of test function is correct before using affine transform
[~, J_actual] = transformTestFunc(mux);
addpath('./DERIVESTsuite');
[J_expected, err] = jacobianest(@(x) transformTestFunc(x), mux);
rmpath('./DERIVESTsuite');
assumeEqual(testCase, J_actual, J_expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected Jacobian to agree with numerical solution');

% Affine transform
py = px.affineTransform(@transformTestFunc);
muy_expected = [0.321750554396642; 3.16227766016838];
verifyEqual(testCase, py.mean(), muy_expected, 'Expected mean to match oracle', 'AbsTol', 1e-10);
Py_expected = [0.000842, 0.00118269184490297; 0.00118269184490297, 0.00178];
verifyEqual(testCase, py.cov(), Py_expected, 'Expected covariance to match oracle', 'AbsTol', 1e-10);
end

function testGaussianTransformUnscented(testCase)
% Input Gaussian
mux = [1; 3];
Sx = [0.1, 0.01; 0, 0.01];
assumeTrue(testCase, istriu(Sx), 'Assume Sx is upper triangular');
px = Gaussian.fromSqrtMoment(mux, Sx);

% Unscented transform
py = px.unscentedTransform(@transformTestFunc);
muy_expected = [0.321377060737193; 3.1636088688768];
verifyEqual(testCase, py.mean(), muy_expected, 'Expected mean to match oracle', 'AbsTol', 1e-10);
Py_expected = [0.000842047795255732, 0.00117992583358464; 0.00117992583358464, 0.00178246899733241];
verifyEqual(testCase, py.cov(), Py_expected, 'Expected covariance to match oracle', 'AbsTol', 1e-10);
end

%% Helper functions
function [f, J] = transformTestFunc(x)
r = hypot(x(1), x(2));
f = [atan2(x(1), x(2)); r];
if nargout > 1
    r2 = r^2;
    J = [x(2)/r2, -x(1)/r2;
        x(1)/r, x(2)/r];
end
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
