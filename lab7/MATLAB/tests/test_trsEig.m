%% Main function to generate tests
function tests = test_trsEig
tests = functiontests(localfunctions);
end

%% Test Functions

function testNewtonStepInsideTrustRegion(testCase)
g = [1; 1; 1; 1];
H = diag(1:4);
[Q, V] = eig(H);
v = diag(V);

Delta = 2;          % Trust region radius
pNewton = -H\g;     % Newton step
assumeLessThanOrEqual(testCase, norm(pNewton), Delta);   % Want ||p|| <= Delta

[p_actual, ret_actual] = funcmin.trsEig(Q, v, g, Delta);
assertEqual(testCase, ret_actual, 0, 'Expected ret = 0');
assertEqual(testCase, p_actual, pNewton, 'Expected Newton step when it is inside trust region', 'AbsTol', 1e-12);
end

function testNewtonStepOutsideTrustRegion(testCase)
g = [1; 1; 1; 1];
H = diag(1:4);
[Q, V] = eig(H);
v = diag(V);

pNewton = -H\g;             % Newton step
Delta = 0.9*norm(pNewton);  % Trust region radius
assumeGreaterThan(testCase, norm(pNewton), Delta);       % Want ||p|| > Delta

[p_actual, ret_actual] = funcmin.trsEig(Q, v, g, Delta);
assertEqual(testCase, ret_actual, 0, 'Expected ret = 0');
assertEqual(testCase, norm(p_actual), Delta, 'Expected step length to equal trust region radius when Newton step is outside trust region', 'AbsTol', 1e-12);
end

function testNonconvex(testCase)
g = [1; 1; 1; 1];
H = diag([-2, -1, 0, 1]);
[Q, V] = eig(H);
v = diag(V);

Delta = 2;          % Trust region radius

[p_actual, ret_actual] = funcmin.trsEig(Q, v, g, Delta);
assertEqual(testCase, ret_actual, 0, 'Expected ret = 0');
assertEqual(testCase, norm(p_actual), Delta, 'Expected step length to equal trust region radius when nonconvex', 'AbsTol', 1e-12);
end

function testNonconvexHardCase(testCase)
g = [0; 1; 1; 1];
H = diag([-2, -1, 0, 1]);
[Q, V] = eig(H);
v = diag(V);

Delta = 2;          % Trust region radius

[p_actual, ret_actual] = funcmin.trsEig(Q, v, g, Delta);
assertEqual(testCase, ret_actual, 0, 'Expected ret = 0');
assertEqual(testCase, norm(p_actual), Delta, 'Expected step length to equal trust region radius when nonconvex (hard case)', 'AbsTol', 1e-12);
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