%% Main function to generate tests
function tests = test_trsSqrt
tests = functiontests(localfunctions);
end

%% Test Functions

function testNewtonStepInsideTrustRegion(testCase)
g = [1; 1; 1; 1];
H = diag(1:4);
Xi = chol(H);

Delta = 2;          % Trust region radius
pNewton = -H\g;     % Newton step
assumeLessThanOrEqual(testCase, norm(Xi*pNewton), Delta);   % Want ||Xi*pNewton|| <= Delta

[p_actual, ret_actual] = funcmin.trsSqrt(Xi, g, Delta);
assertEqual(testCase, ret_actual, 0, 'Expected ret = 0');
p_expected = pNewton;
assertEqual(testCase, p_actual, p_expected, 'Expected Newton step when it is inside trust region', 'AbsTol', 1e-12);
end

function testNewtonStepOutsideTrustRegion(testCase)
g = [1; 1; 1; 1];
H = diag(1:4);
Xi = chol(H);

pNewton = -H\g;
Delta = 0.9*norm(pNewton);   % Trust region radius
assumeGreaterThan(testCase, norm(Xi*pNewton), Delta);       % Want ||Xi*pNewton|| > Delta

[p_actual, ret_actual] = funcmin.trsSqrt(Xi, g, Delta);
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