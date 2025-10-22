% Minimise f(x) using trust-region quasi-Newton BFGS method (square-root inverse Hessian)
function [x, g, S, ret] = BFGSTrustSqrtInv(costFunc, x, S, verbosity)

arguments
    costFunc (1, 1) function_handle
    x (:, 1) double
    S (:, :) double = eye(length(x))
    verbosity (1, 1) uint8 = 0
end

assert(istriu(S));

% Cache structure used by linsolve
persistent s_ut_transa
if isempty(s_ut_transa)
    s_ut_transa = struct('UT', true, 'TRANSA', true);
end

nx = length(x);

% Evaluate initial cost and gradient
[f, g] = costFunc(x);
if ~isfinite(f) || ~all(isfinite(g))            % if any nan, -inf or +inf
    if (verbosity > 1)
        fprintf(2, 'ERROR: Initial point is not in domain of cost function\n');
    end
    ret = -1;
    return
end

Delta = 10e0;     % Initial trust-region radius

maxIterations = 5000;
for i = 1:maxIterations
    % Solve trust-region subproblem
    p = funcmin.trsSqrtInv(S, g, Delta);    % minimise 0.5*p.'*inv(S.'*S)*p + g.'*p subject to ||inv(S.')*p|| <= Delta
    % Solve S.'*z = p for z
    z = linsolve(S, p, s_ut_transa); % z = S.'\p
    
    pg = p.'*g;
    LambdaSq = -pg;  % The Newton decrement squared is g.'*inv(H)*g = p.'*H*p = p.'*inv(S.'*S)*p
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
    
    % Evaluate cost and gradient for trial step
    xn = x + p;
    [fn, gn] = costFunc(xn);
    outOfDomain = ~isfinite(fn) || ~all(isfinite(gn));      % if any nan, -inf or +inf
    y = gn - g;
    
    if outOfDomain
        rho = -1;                           % Force trust region reduction and reject step
    else
        dm = -pg - 0.5*(z.'*z);             % Predicted reduction f - fp, where fp = f + p'*g + 0.5*p'*H*p
        rho = (f - fn)/dm;                  % Actual reduction divided by predicted reduction
    end
 
    if rho < 0.1
        Delta = 0.25*norm(z);               % Decrease trust region radius
    else
        if rho > 0.75 && norm(z) > 0.8*Delta
            Delta = 2.0*Delta;              % Increase trust region radius
        end
    end
    
    if rho >= 0.001
        % Accept the step
        x = xn;
        f = fn;
        g = gn;
    end
    
    % Update Hessian approximation
    sqrteps = realsqrt(eps);
    if ~outOfDomain
        py = p.'*y;
        if (py > sqrteps*norm(y)*norm(p))
            % Form
            % [       p^T/sqrt(y^T*p) ]
            % [ S*(I - y*p^T/(y^T*p)) ]
            RR = [ p.'/realsqrt(py);
                   S*(eye(nx) - y*p.'/py)];
            % Q-less QR
            S = qr(RR, "econ");
        end
    end
end
if verbosity > 1
    fprintf(2, 'WARNING: maximum number of iterations reached\n');
end
ret = 1;
return
