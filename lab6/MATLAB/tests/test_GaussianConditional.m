%% Main function to generate tests
function tests = test_GaussianConditional
tests = functiontests(localfunctions);
end

%% Test Functions

function testConditional43Head(testCase)
mu =[1;
     1;
     1;
     1];
S = [ -0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069;
                       0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469;
                       0,                   0,  -0.196861446475943,    0.800320709801823;
                       0,                   0,                   0,    -1.50940472473439];
p = Gaussian.fromSqrtMoment(mu, S);
xB = [0.875874147834533;
      -0.24278953633334;
      0.166813439453503];
idxA = 4;
idxB = [1, 2, 3];
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional43Tail(testCase)
mu =[1;
     1;
     1;
     1];
S = [ -0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069;
                       0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469;
                       0,                   0,  -0.196861446475943,    0.800320709801823;
                       0,                   0,                   0,    -1.50940472473439];
p = Gaussian.fromSqrtMoment(mu, S);
xB = [0.875874147834533;
      -0.24278953633334;
      0.166813439453503];
idxA = 1;
idxB = [2, 3, 4];
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional41Head(testCase)
mu =[1;
     1;
     1;
     1];
S = [ -0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069;
                       0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469;
                       0,                   0,  -0.196861446475943,    0.800320709801823;
                       0,                   0,                   0,    -1.50940472473439];
p = Gaussian.fromSqrtMoment(mu, S);
xB = 0.875874147834533;
idxA = [2, 3, 4];
idxB = 1;
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional41Tail(testCase)
mu =[1;
     1;
     1;
     1];
S = [ -0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069;
                       0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469;
                       0,                   0,  -0.196861446475943,    0.800320709801823;
                       0,                   0,                   0,    -1.50940472473439];
p = Gaussian.fromSqrtMoment(mu, S);
xB = 0.875874147834533;
idxA = [1, 2, 3];
idxB = 4;
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional41Segment(testCase)
mu =[1;
     1;
     1;
     1];
S = [ -0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069;
                       0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469;
                       0,                   0,  -0.196861446475943,    0.800320709801823;
                       0,                   0,                   0,    -1.50940472473439];
p = Gaussian.fromSqrtMoment(mu, S);
xB = 0.875874147834533;
idxA = [1, 2, 4];
idxB = 3;
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional63Head(testCase)
mu = [1;
      1;
      1;
      1;
      1;
      1];
S = [-0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069,   -1.50940472473439,   0.166813439453503;
                      0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469,   0.875874147834533,   -1.96541870928278;
                      0,                   0,  -0.196861446475943,    0.800320709801823,   -0.24278953633334,   -1.27007139263854;
                      0,                   0,                   0,     1.17517126546302,   0.603658445825815,   -1.86512257453063;
                      0,                   0,                   0,                    0,     1.7812518932425,   -1.05110705924059;
                      0,                   0,                   0,                    0,                   0,  -0.417382047996795];
p = Gaussian.fromSqrtMoment(mu, S);
xB = [0.875874147834533;
      -0.24278953633334;
      0.166813439453503];
idxA = [4, 5, 6];
idxB = [1, 2, 3];
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional63Tail(testCase)
mu = [1;
      1;
      1;
      1;
      1;
      1];
S = [-0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069,   -1.50940472473439,   0.166813439453503;
                      0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469,   0.875874147834533,   -1.96541870928278;
                      0,                   0,  -0.196861446475943,    0.800320709801823,   -0.24278953633334,   -1.27007139263854;
                      0,                   0,                   0,     1.17517126546302,   0.603658445825815,   -1.86512257453063;
                      0,                   0,                   0,                    0,     1.7812518932425,   -1.05110705924059;
                      0,                   0,                   0,                    0,                   0,  -0.417382047996795];
p = Gaussian.fromSqrtMoment(mu, S);
xB = [0.875874147834533;
      -0.24278953633334;
      0.166813439453503];
idxA = [1, 2, 3];
idxB = [4, 5, 6];
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional63Segment(testCase)
mu = [1;
      1;
      1;
      1;
      1;
      1];
S = [-0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069,   -1.50940472473439,   0.166813439453503;
                      0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469,   0.875874147834533,   -1.96541870928278;
                      0,                   0,  -0.196861446475943,    0.800320709801823,   -0.24278953633334,   -1.27007139263854;
                      0,                   0,                   0,     1.17517126546302,   0.603658445825815,   -1.86512257453063;
                      0,                   0,                   0,                    0,     1.7812518932425,   -1.05110705924059;
                      0,                   0,                   0,                    0,                   0,  -0.417382047996795];
p = Gaussian.fromSqrtMoment(mu, S);
xB = [0.875874147834533;
      -0.24278953633334;
      0.166813439453503];
idxA = [1, 2, 6];
idxB = [3, 4, 5];
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional63Noncontiguous(testCase)
mu = [1;
      1;
      1;
      1;
      1;
      1];
S = [-0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069,   -1.50940472473439,   0.166813439453503;
                      0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469,   0.875874147834533,   -1.96541870928278;
                      0,                   0,  -0.196861446475943,    0.800320709801823,   -0.24278953633334,   -1.27007139263854;
                      0,                   0,                   0,     1.17517126546302,   0.603658445825815,   -1.86512257453063;
                      0,                   0,                   0,                    0,     1.7812518932425,   -1.05110705924059;
                      0,                   0,                   0,                    0,                   0,  -0.417382047996795];
p = Gaussian.fromSqrtMoment(mu, S);
xB = [0.875874147834533;
      -0.24278953633334;
      0.166813439453503];
idxA = [1, 3, 5];
idxB = [2, 4, 6];
runConditionalCase(testCase, p, idxA, idxB, xB);
end

function testConditional63NoncontiguousNonascending(testCase)
mu = [1;
      1;
      1;
      1;
      1;
      1];
S = [-0.649013765191241,   -1.10961303850152,  -0.558680764473972,    0.586442621667069,   -1.50940472473439,   0.166813439453503;
                      0,  -0.845551240007797,   0.178380225849766,   -0.851886969622469,   0.875874147834533,   -1.96541870928278;
                      0,                   0,  -0.196861446475943,    0.800320709801823,   -0.24278953633334,   -1.27007139263854;
                      0,                   0,                   0,     1.17517126546302,   0.603658445825815,   -1.86512257453063;
                      0,                   0,                   0,                    0,     1.7812518932425,   -1.05110705924059;
                      0,                   0,                   0,                    0,                   0,  -0.417382047996795];
p = Gaussian.fromSqrtMoment(mu, S);
xB = [0.875874147834533;
      -0.24278953633334;
      0.166813439453503];
idxA = [3, 1, 5];
idxB = [2, 6, 4];
runConditionalCase(testCase, p, idxA, idxB, xB);
end


function testConditionalCovarianceOverflow(testCase)
n = 2;
mu = zeros(n, 1);
S = 1e300*eye(n);
assumeTrue(testCase, istriu(S), 'Assume S is upper triangular');
p = Gaussian.fromSqrtMoment(mu, S);
assumeFalse(testCase, all(isfinite(p.cov()), 'all'), 'Assume covariance overflows to inf')
idxA = 2;
idxB = 1;
xB = 0;
pc = p.conditional(idxA, idxB, xB);
assertTrue(testCase, all(isfinite(pc.sqrtCov()), 'all'), 'Expected marginal square-root covariance to be finite')
end

% Helper functions
function runConditionalCase(testCase, p, idxA, idxB, xB)
    % Condition on a value
    runConditionalCaseValue(testCase, p, idxA, idxB, xB);

    % Condition on a delta distribution
    pB = Gaussian.fromSqrtMoment(xB, zeros(length(xB)));
    runConditionalCaseDist(testCase, p, idxA, idxB, pB);

    % Condition on a Gaussian distribution
    pB = Gaussian.fromSqrtMoment(xB, triu(magic(length(xB))));
    runConditionalCaseDist(testCase, p, idxA, idxB, pB);
end

function runConditionalCaseValue(testCase, p, idxA, idxB, xB)
mu = p.mean();
S = p.sqrtCov();
assumeTrue(testCase, istriu(S), 'Expected S to be upper triangular');

pc = p.conditional(idxA, idxB, xB); % Given data is a value
muc_actual = pc.mean();
Sc_actual = pc.sqrtCov();
assertTrue(testCase, istriu(Sc_actual), 'Expected conditional sqrt cov to be upper triangular');

P = p.cov(); % S.'*S
muc_expected = mu(idxA) + P(idxA, idxB)*( P(idxB, idxB)\(xB - mu(idxB)) );
assertEqual(testCase, muc_actual, muc_expected, 'Expected conditional mean to match expected result', 'AbsTol', 1e-12);

Pc_actual = pc.cov(); % Sc_actual.'*Sc_actual
Pc_expected = P(idxA, idxA) - P(idxA, idxB)*(P(idxB, idxB)\P(idxB, idxA));
assertEqual(testCase, Pc_actual, Pc_expected, 'Expected conditional cov to match expected result', 'AbsTol', 1e-10);
end

function runConditionalCaseDist(testCase, p, idxA, idxB, pB)
mu = p.mean();
S = p.sqrtCov();
assumeTrue(testCase, istriu(S), 'Expected S to be upper triangular');

pc = p.conditional(idxA, idxB, pB); % Given data is a distribution
muc_actual = pc.mean();
Sc_actual = pc.sqrtCov();
assertTrue(testCase, istriu(Sc_actual), 'Expected conditional sqrt cov to be upper triangular');

P = p.cov(); % S.'*S
muc_expected = mu(idxA) + P(idxA, idxB)*( P(idxB, idxB)\(pB.mean() - mu(idxB)) );
assertEqual(testCase, muc_actual, muc_expected, 'Expected conditional mean to match expected result', 'AbsTol', 1e-12);

Pc_actual = pc.cov(); % Sc_actual.'*Sc_actual
Pc_expected = P(idxA, idxA) + (P(idxA, idxB)/P(idxB, idxB))*(pB.cov() - P(idxB, idxB))*(P(idxB, idxB)\P(idxB, idxA));
assertEqual(testCase, Pc_actual, Pc_expected, 'Expected conditional cov to match expected result', 'AbsTol', 1e-10);
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
