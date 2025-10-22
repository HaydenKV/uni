%% Main function to generate tests
function tests = test_trsSqrtInv
tests = functiontests(localfunctions);
end

%% Test Functions

function testNewtonStepInsideTrustRegion(testCase)
g = [1; 1; 1; 1];
H = diag(1:4);
Xi = chol(H);
S = Xi.'\eye(4);
S = qr(S, "econ");

Delta = 2;          % Trust region radius
gtilde = S*g;
assumeLessThanOrEqual(testCase, norm(gtilde), Delta);   % Want ||S*g|| <= Delta

[p_actual, ret_actual] = funcmin.trsSqrtInv(S, g, Delta);
assertEqual(testCase, ret_actual, 0, 'Expected ret = 0');
p_expected = -H\g;  % Newton step
assertEqual(testCase, p_actual, p_expected, 'Expected Newton step when it is inside trust region', 'AbsTol', 1e-12);
end

function testNewtonStepOutsideTrustRegion(testCase)
g = [1; 1; 1; 1];
H = diag(1:4);
Xi = chol(H);
S = Xi.'\eye(4);
S = qr(S, "econ");

gtilde = S*g;
Delta = 0.9*norm(gtilde);   % Trust region radius
assumeGreaterThan(testCase, norm(gtilde), Delta);       % Want ||S*g|| > Delta

[p_actual, ret_actual] = funcmin.trsSqrtInv(S, g, Delta);
assertEqual(testCase, ret_actual, 0, 'Expected ret = 0');
assertEqual(testCase, norm(Xi*p_actual), Delta, 'Expected step length to equal trust region radius when Newton step is outside trust region', 'AbsTol', 1e-12);
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