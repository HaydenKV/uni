%% Main function to generate tests
function tests = test_GaussianInfo
tests = functiontests(localfunctions);
end

%% Test Functions
function testZeroMean(testCase)
mu = zeros(5, 1);
S = [ ...
    10, 11, 12, 13, 14; ...
     0, 15, 16, 17, 18; ...
     0,  0, 19, 20, 21; ...
     0,  0,  0, 23, 24; ...
     0,  0,  0,  0, 25];
runGaussianInfoCases(testCase, mu, S);
end

function testNonZeroMean(testCase)
mu = [1; 2; 3; 4; 5];
S = [ ...
    10, 11, 12, 13, 14; ...
     0, 15, 16, 17, 18; ...
     0,  0, 19, 20, 21; ...
     0,  0,  0, 23, 24; ...
     0,  0,  0,  0, 25];
runGaussianInfoCases(testCase, mu, S);
end

%% Helper functions
function runGaussianInfoCases(testCase, mu, S)
P = S.'*S;
% Xi = qr(S^{-T})
Xi = qr(S.'\eye(size(S, 2)), "econ");
Lambda = Xi.'*Xi;

nu = Xi*mu;
eta = Lambda*mu;

p = GaussianInfo.fromSqrtMoment(mu, S);
runGaussianInfoCase(testCase, p, mu, P, eta, Lambda);

p = GaussianInfo.fromMoment(mu, P);
runGaussianInfoCase(testCase, p, mu, P, eta, Lambda);

p = GaussianInfo.fromInfo(eta, Lambda);
runGaussianInfoCase(testCase, p, mu, P, eta, Lambda);

p = GaussianInfo.fromSqrtInfo(nu, Xi);
runGaussianInfoCase(testCase, p, mu, P, eta, Lambda);
end

function runGaussianInfoCase(testCase, p, mu_expected, P_expected, eta_expected, Lambda_expected)
n = p.dim();

mu_actual = p.mean();
verifyEqual(testCase, size(mu_actual), [n, 1], 'Expected mu to have correct dimensions');
verifyEqual(testCase, mu_actual, mu_expected, 'Expected mu to match oracle', 'AbsTol', 1e-10);

eta_actual = p.infoVec();
verifyEqual(testCase, size(eta_actual), [n, 1], 'Expected eta to have correct dimensions');
verifyEqual(testCase, eta_actual, eta_expected, 'Expected eta to match oracle', 'AbsTol', 1e-10);

Xi_actual = p.sqrtInfoMat();
verifyEqual(testCase, size(Xi_actual), [n, n], 'Expected Xi to have correct dimensions');
verifyTrue(testCase, istriu(Xi_actual), 'Expected Xi to be upper triangular');

nu_actual = p.sqrtInfoVec();
verifyEqual(testCase, size(nu_actual), [n, 1], 'Expected nu to have correct dimensions');

P_actual = p.cov();
verifyEqual(testCase, size(P_actual), [n, n], 'Expected P to have correct dimensions');
verifyEqual(testCase, P_actual, P_expected, 'Expected P to match oracle', 'AbsTol', 1e-10);

S_actual = p.sqrtCov();
verifyEqual(testCase, size(S_actual), [n, n], 'Expected S to have correct dimensions');
verifyTrue(testCase, istriu(S_actual), 'Expected S to be upper triangular');

Lambda_actual = p.infoMat();
verifyEqual(testCase, size(Lambda_actual), [n, n], 'Expected Lambda to have correct dimensions');
verifyEqual(testCase, Lambda_actual, Lambda_expected, 'Expected Lambda to match oracle', 'AbsTol', 1e-10);

verifyEqual(testCase, S_actual.'*S_actual, P_actual, "Expected S.'*S = P", 'AbsTol', 1e-10);
verifyEqual(testCase, Xi_actual.'*Xi_actual, Lambda_actual, "Expected Xi.'*Xi = Lambda", 'AbsTol', 1e-10);
verifyEqual(testCase, Lambda_actual*mu_actual, eta_actual, 'Expected Lambda*mu = eta', 'AbsTol', 1e-10);
verifyEqual(testCase, P_actual*eta_actual, mu_actual, 'Expected P*eta = mu', 'AbsTol', 1e-10);
verifyEqual(testCase, Xi_actual*mu_actual, nu_actual, 'Expected Xi*mu = nu', 'AbsTol', 1e-10);
verifyEqual(testCase, Xi_actual.'*nu_actual, eta_actual, "Expected Xi.'*nu = eta", 'AbsTol', 1e-10);
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