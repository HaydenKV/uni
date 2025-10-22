%% Main function to generate tests
function tests = test_GaussianInfoConfidence
    tests = functiontests(localfunctions);
end

%% Test Functions
function testGaussianConfidenceIdentityS(testCase)
mu = [0.5; 1.5];
S = eye(2);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = GaussianInfo.fromSqrtMoment(mu, S);
n_sigma = 3;
n_samples = 100;
X = p.confidenceEllipse(n_sigma, n_samples);
m = mean(X(:, 1:end-1), 2); % Remove either start or end point, since they overlap
verifyEqual(testCase, m, mu, 'Expected points centred on mean', 'AbsTol', 1e-10);
for i = 1:n_samples
    verifyFalse(testCase, p.isWithinConfidenceRegion(X(:, i), 2.9), 'Expected point on 3 sigma ellipse to be outside 2.9 sigma confidence region');
    verifyTrue(testCase, p.isWithinConfidenceRegion(X(:, i), 3.1), 'Expected point on 3 sigma ellipse to be inside 3.1 sigma confidence region');
end
end

function testGaussianConfidenceZeroMean(testCase)
mu = zeros(2, 1);
S = [1, 0.3; 0, 0.1];
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = GaussianInfo.fromSqrtMoment(mu, S);
n_sigma = 3;
n_samples = 100;
X = p.confidenceEllipse(n_sigma, n_samples);
W = S.'\X;
r = vecnorm(W, 2, 1);
verifyEqual(testCase, r, repmat(3.43935431177144, size(r)), 'Expected all points lie on the confidence region boundary', 'AbsTol', 1e-10);
for i = 1:n_samples
    verifyFalse(testCase, p.isWithinConfidenceRegion(X(:, i), 2.9), 'Expected point on 3 sigma ellipse to be outside 2.9 sigma confidence region');
    verifyTrue(testCase, p.isWithinConfidenceRegion(X(:, i), 3.1), 'Expected point on 3 sigma ellipse to be inside 3.1 sigma confidence region');
end
end

function testGaussianConfidenceQuadricZeroMeanIdentityS(testCase)
mu = zeros(3, 1);
S = eye(3);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = GaussianInfo.fromSqrtMoment(mu, S);
n_sigma = 3;
Q_actual = p.quadricSurface(n_sigma);
Q_expected = diag([1, 1, 1, -14.1564136091267]);
assertEqual(testCase, Q_actual, Q_expected, 'Expected Q to match oracle', 'AbsTol', 1e-10);
end

function testGaussianConfidenceQuadricDiagS(testCase)
mu = [1; 2; 3];
S = diag([4, 5, 6]);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = GaussianInfo.fromSqrtMoment(mu, S);
n_sigma = 3;
Q_actual = p.quadricSurface(n_sigma);
Q_expected = [1.0/16.0,         0,         0,         -1.0/16.0;
                     0,  1.0/25.0,         0,         -2.0/25.0;
                     0,         0,  1.0/36.0,         -1.0/12.0;
             -1.0/16.0, -2.0/25.0, -1.0/12.0, -13.6839136091267];
assertEqual(testCase, Q_actual, Q_expected, 'Expected Q to match oracle', 'AbsTol', 1e-10);
end

function testGaussianConfidenceQuadricUTS(testCase)
mu = [1; 2; 3];
S = [-0.6490137652,      -1.109613039,     -0.5586807645;
                 0,       -0.84555124,      0.1783802258;
                 0,                 0,     -0.1968614465];
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = GaussianInfo.fromSqrtMoment(mu, S);
n_sigma = 3;
Q_actual = p.quadricSurface(n_sigma);
Q_expected = [44.9627201133221,  -9.0406493633664, -31.5188988229375,  67.6752750822232;
              -9.0406493633664,  2.54708314580204,  5.44359034070936, -12.3842879503658;
             -31.5188988229375,  5.44359034070936,   25.803502277206, -56.7787886900993;
              67.6752750822232, -12.3842879503658, -56.7787886900993,   113.27325327968];
assertEqual(testCase, Q_actual, Q_expected, 'Expected Q to match oracle', 'AbsTol', 1e-10);
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
