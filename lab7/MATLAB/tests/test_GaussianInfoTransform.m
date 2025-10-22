%% Main function to generate tests
function tests = test_GaussianInfoTransform
    tests = functiontests(localfunctions);
end

%% Test Functions
function testGaussianInfoTransformAffineSquareFullRank(testCase)
mux = [1; 3];
Sx = [0.1, 0.01; 0, 0.01];
assumeTrue(testCase, istriu(Sx), 'Assume Sx is upper triangular');
px = GaussianInfo.fromSqrtMoment(mux, Sx);

% Assume Jacobian of test function is correct before using affine transform
[~, J_actual] = transformTestFuncSquareFullRank(mux);
addpath('./DERIVESTsuite');
[J_expected, err] = jacobianest(@(x) transformTestFuncSquareFullRank(x), mux);
rmpath('./DERIVESTsuite');
assumeEqual(testCase, J_actual, J_expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected Jacobian to agree with numerical solution');

% Affine transform
py = px.affineTransform(@transformTestFuncSquareFullRank);
muy_expected = [0.321750554396642; 3.16227766016838];
verifyEqual(testCase, py.mean(), muy_expected, 'Expected mean to match oracle', 'AbsTol', 1e-10);
Py_expected = [0.000842, 0.00118269184490297; 0.00118269184490297, 0.00178];
verifyEqual(testCase, py.cov(), Py_expected, 'Expected covariance to match oracle', 'AbsTol', 1e-10);
end

function testGaussianInfoTransformAffineWide(testCase)
mux = [1; 3; 5];
Sx = [0.1, 0.01, 0; 0, 0.01, 0; 0, 0, 1];
assumeTrue(testCase, istriu(Sx), 'Assume Sx is upper triangular');
px = GaussianInfo.fromSqrtMoment(mux, Sx);

% Assume Jacobian of test function is correct before using affine transform
[~, J_actual] = transformTestFuncWide(mux);
addpath('./DERIVESTsuite');
[J_expected, err] = jacobianest(@(x) transformTestFuncWide(x), mux);
rmpath('./DERIVESTsuite');
assumeEqual(testCase, J_actual, J_expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected Jacobian to agree with numerical solution');

% Affine transform
py = px.affineTransform(@transformTestFuncWide);
muy_expected = [0.321750554396642; 3.16227766016838];
verifyEqual(testCase, py.mean(), muy_expected, 'Expected mean to match oracle', 'AbsTol', 1e-10);
Py_expected = [0.000842, 0.00118269184490297; 0.00118269184490297, 0.00178];
verifyEqual(testCase, py.cov(), Py_expected, 'Expected covariance to match oracle', 'AbsTol', 1e-10);
end

function testGaussianInfoTransformAffineWideLowRank(testCase)
mux = [1; 3; 5];
Sx = [0.1, 0.01, 0; 0, 0.01, 0; 0, 0, 1];
assumeTrue(testCase, istriu(Sx), 'Assume Sx is upper triangular');
px = GaussianInfo.fromSqrtMoment(mux, Sx);

% Assume Jacobian of test function is correct before using affine transform
[~, J_actual] = transformTestFuncWideLowRank(mux);
addpath('./DERIVESTsuite');
[J_expected, err] = jacobianest(@(x) transformTestFuncWideLowRank(x), mux);
rmpath('./DERIVESTsuite');
assumeEqual(testCase, J_actual, J_expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected Jacobian to agree with numerical solution');

% Affine transform
py = px.affineTransform(@transformTestFuncWideLowRank).marginal(2);
muy_expected = 3.16227766016838;
verifyEqual(testCase, py.mean(), muy_expected, 'Expected mean to match oracle', 'AbsTol', 1e-7);
Py_expected = 0.00178;
verifyEqual(testCase, py.cov(), Py_expected, 'Expected covariance to match oracle', 'AbsTol', 1e-7);
end

function testGaussianInfoTransformAffineWide1Row(testCase)
mux = [1; 3];
Sx = [0.1, 0.01; 0, 0.01];
assumeTrue(testCase, istriu(Sx), 'Assume Sx is upper triangular');
px = GaussianInfo.fromSqrtMoment(mux, Sx);

% Assume Jacobian of test function is correct before using affine transform
[~, J_actual] = transformTestFuncWide1Row(mux);
addpath('./DERIVESTsuite');
[J_expected, err] = jacobianest(@(x) transformTestFuncWide1Row(x), mux);
rmpath('./DERIVESTsuite');
assumeEqual(testCase, J_actual, J_expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected Jacobian to agree with numerical solution');

% Affine transform
py = px.affineTransform(@transformTestFuncWide1Row);
muy_expected = 3.16227766016838;
verifyEqual(testCase, py.mean(), muy_expected, 'Expected mean to match oracle', 'AbsTol', 1e-10);
Py_expected = 0.00178;
verifyEqual(testCase, py.cov(), Py_expected, 'Expected covariance to match oracle', 'AbsTol', 1e-10);
end

function testGaussianInfoTransformAffineTall(testCase)
mux = [1; 3];
Sx = [0.1, 0.01; 0, 0.01];
assumeTrue(testCase, istriu(Sx), 'Assume Sx is upper triangular');
px = GaussianInfo.fromSqrtMoment(mux, Sx);

% Assume Jacobian of test function is correct before using affine transform
[~, J_actual] = transformTestFuncTall(mux);
addpath('./DERIVESTsuite');
[J_expected, err] = jacobianest(@(x) transformTestFuncTall(x), mux);
rmpath('./DERIVESTsuite');
assumeEqual(testCase, J_actual, J_expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected Jacobian to agree with numerical solution');

% Affine transform
py = px.affineTransform(@transformTestFuncTall).marginal(1:2);
muy_expected = [0.321750554396642; 3.16227766016838];
verifyEqual(testCase, py.mean(), muy_expected, 'Expected mean to match oracle', 'AbsTol', 1e-7);
Py_expected = [0.000842, 0.00118269184490297; 0.00118269184490297, 0.00178];
verifyEqual(testCase, py.cov(), Py_expected, 'Expected covariance to match oracle', 'AbsTol', 1e-7);
end

function testGaussianInfoTransformAffineSquareLowRank(testCase)
mux = [1; 3];
Sx = [0.1, 0.01; 0, 0.01];
assumeTrue(testCase, istriu(Sx), 'Assume Sx is upper triangular');
px = GaussianInfo.fromSqrtMoment(mux, Sx);

% Assume Jacobian of test function is correct before using affine transform
[~, J_actual] = transformTestFuncSquareLowRank(mux);
addpath('./DERIVESTsuite');
[J_expected, err] = jacobianest(@(x) transformTestFuncSquareLowRank(x), mux);
rmpath('./DERIVESTsuite');
assumeEqual(testCase, J_actual, J_expected, 'AbsTol', max(100*err, 1e-10), ...
    'Expected Jacobian to agree with numerical solution');

% Affine transform
py = px.affineTransform(@transformTestFuncSquareLowRank);
muy_expected = [3.16227766016838; 6.32455531152522];
verifyEqual(testCase, py.mean(), muy_expected, 'Expected mean to match oracle', 'AbsTol', 1e-7);
Py_expected = [0.00178, 0.00356; 0.00356, 0.00712];
verifyEqual(testCase, py.cov(), Py_expected, 'Expected covariance to match oracle', 'AbsTol', 1e-7);
end

%% Helper functions
function [f, J] = transformTestFuncSquareFullRank(x)
r = hypot(x(1), x(2));
f = [atan2(x(1), x(2)); r];
if nargout > 1
    r2 = r^2;
    J = [x(2)/r2, -x(1)/r2;
        x(1)/r, x(2)/r];
end
end

function [f, J] = transformTestFuncTall(x)
r = hypot(x(1), x(2));
theta = atan2(x(1), x(2));
f = [theta; r; 2*r];
if nargout > 1
    r2 = r^2;
    J = [x(2)/r2, -x(1)/r2;
        x(1)/r, x(2)/r;
        2*x(1)/r, 2*x(2)/r];
end
end

function [f, J] = transformTestFuncWide(x)
r = hypot(x(1), x(2));
f = [atan2(x(1), x(2)); r];
if nargout > 1
    r2 = r^2;
    J = [x(2)/r2, -x(1)/r2, 0;
        x(1)/r, x(2)/r, 0];
end
end

function [f, J] = transformTestFuncWideLowRank(x)
r = hypot(x(1), x(2));
f = [2*r; r];
if nargout > 1
    J = [2*x(1)/r, 2*x(2)/r, 0;
        x(1)/r, x(2)/r, 0];
end
end

function [f, J] = transformTestFuncWide1Row(x)
f = hypot(x(1), x(2));
if nargout > 1
    J = [x(1)/f, x(2)/f]; % Wide: m < n
end
end

function [f, J] = transformTestFuncSquareLowRank(x)
r = hypot(x(1), x(2));
f = [r; 2*r];
if nargout > 1
    J = [x(1)/r, x(2)/r;
        2*x(1)/r, 2*x(2)/r];
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
