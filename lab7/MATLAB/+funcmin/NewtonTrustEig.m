% Minimise f(x) using trust-region Newton method (Eigendecomposition of Hessian)
function [x, g, Q, v, ret] = NewtonTrustEig(costFunc, x, verbosity)

arguments
    costFunc (1, 1) function_handle
    x (:, 1) double
    verbosity (1, 1) uint8 = 0
end

% Evaluate initial cost and gradient
[f, g, H] = costFunc(x);
if ~isfinite(f) || ~all(isfinite(g)) || ~all(isfinite(H), 'all')    % if any nan, -inf or +inf
    if (verbosity > 1)
        fprintf(2, 'ERROR: Initial point is not in domain of cost function\n');
    end
    ret = -1;
    return
end

% Eigendecomposition of initial Hessian
[Q, v] = eig(H, 'vector');

Delta = 10e0;     % Initial trust-region radius

maxIterations = 5000;
for i = 1:maxIterations
    % Solve trust-region subproblem
    p = funcmin.trsEig(Q, v, g, Delta);     % minimise 0.5*p.'*H*p + g.'*p subject to ||p|| <= Delta
    
    pg = p.'*g;
    LambdaSq = -pg;  % The Newton decrement squared is g.'*inv(H)*g = p.'*H*p
    if verbosity == 3
        fprintf(1, 'Iter = %5i, Cost = %10.2e, Newton decr^2 = %10.2e, Delta = %10.2e\n', i, f, LambdaSq, Delta);
    end
    if verbosity == 1
        fprintf(1, '.');
    end
    
    LambdaSqThreshold = 2*eps;
    if abs(LambdaSq) < LambdaSqThreshold
        if (verbosity >= 2)
            fprintf(1, 'CONVERGED: Newton decrement below threshold in %i iterations\n', i);
        end
        ret = 0;
        return
    end
    
    % Evaluate cost, gradient and Hessian for trial step
    xn = x + p;
    [fn, gn, Hn] = costFunc(xn);
    outOfDomain = ~isfinite(fn) || ~all(isfinite(gn)) || ~all(isfinite(Hn), 'all');     % if any nan, -inf or +inf
    % y = gn - g;
    
    if outOfDomain
        rho = -1;                           % Force trust region reduction and reject step
    else
        dm = -pg - 0.5*p.'*H*p;             % Predicted reduction f - fp, where fp = f + p'*g + 0.5*p'*H*p
        rho = (f - fn)/dm;                  % Actual reduction divided by predicted reduction
    end
 
    if rho < 0.1
        Delta = 0.25*norm(p);               % Decrease trust region radius
    else
        if rho > 0.75 && norm(p) > 0.8*Delta
            Delta = 2.0*Delta;              % Increase trust region radius
        end
    end
    
    if rho >= 0.001
        % Accept the step
        x = xn;
        f = fn;
        g = gn;
        H = Hn;

        % Eigendecomposition of accepted Hessian
        [Q, v] = eig(Hn, 'vector');
    end
end
if verbosity > 1
    fprintf(2, 'WARNING: maximum number of iterations reached\n');
end
ret = 1;
return
