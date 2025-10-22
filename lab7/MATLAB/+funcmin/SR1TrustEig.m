% Minimise f(x) using trust-region quasi-Newton SR1 method (Eigendecomposition of Hessian)
function [x, g, Q, v, ret] = SR1TrustEig(costFunc, x, Q, v, verbosity)

arguments
    costFunc (1, 1) function_handle
    x (:, 1) double
    Q (:, :) double
    v (:, 1) double
    verbosity (1, 1) uint8 = 0
end

% Evaluate initial cost and gradient
[f, g] = costFunc(x);
if ~isfinite(f) || ~all(isfinite(g))            % if any nan, -inf or +inf
    if (verbosity > 1)
        fprintf(2, 'ERROR: Initial point is not in domain of cost function\n');
    end
    ret = -1;
    return
end

% Initial Hessian
H = Q*(v.*Q.');

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
    
    LambdaSqThreshold = realsqrt(eps);
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
    end
    
    % Update Hessian approximation
    sqrteps = realsqrt(eps);
    if ~outOfDomain
        w = y - H*p;
        pw = p.'*w;
        if abs(pw) > sqrteps*norm(p)*norm(w)
            s = sign(pw);
            u = w./realsqrt(s*pw);
            H = H + s*(u*u.');

            [Q, v] = eig(H, 'vector');
            % We should do a rank-one update of Q and v instead of a full eigendecomposition using, e.g.,
            % [1] Bunch, J.R., Nielsen, C.P. and Sorensen, D.C., 1978. Rank-one modification of the symmetric eigenproblem. Numerische Mathematik, 31(1), pp.31-48.
        end
    end
end
if verbosity > 1
    fprintf(2, 'WARNING: maximum number of iterations reached\n');
end
ret = 1;
return
