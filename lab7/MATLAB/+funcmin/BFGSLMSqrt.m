% Minimise f(x) using LM quasi-Newton BFGS method (square-root Hessian)
function [x, g, Xi, ret] = BFGSLMSqrt(costFunc, x, Xi, verbosity)

arguments
    costFunc (1, 1) function_handle
    x (:, 1) double
    Xi (:, :) double = eye(length(x))
    verbosity (1, 1) uint8 = 0
end

assert(istriu(Xi));

% Cache structure used by linsolve
persistent s_ut s_ut_transa
if isempty(s_ut) || isempty(s_ut_transa)
    s_ut = struct('UT', true, 'TRANSA', false);
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

lambda = 1e-7;
% lambda = 1e-7*max(sum(Xi.^2, 1));

maxIterations = 5000;
for i = 1:maxIterations
    % Solve the secular equation (Xi.'*Xi + lambda*I)*p = -g for p
    XiI = qr([Xi; realsqrt(lambda)*eye(nx)], "econ");
    XiIg = linsolve(XiI, g, s_ut_transa); % XiI.'\g
    p = -linsolve(XiI, XiIg, s_ut); % -XiI\(XiI.'\g)
    z = Xi*p;
    
    pg = p.'*g;
    NewtonDecrSq = -pg;  % The Newton decrement squared is g.'*inv(H)*g = p.'*H*p if p is the Newton step
    % NewtonDecrSq = z.'*z;
    if verbosity == 3
        fprintf(1, 'Iter = %5i, Cost = %10.2e, Newton decr^2 = %10.2e, Lambda = %10.2e\n', i, f, NewtonDecrSq, lambda);
    end
    if verbosity == 1
        fprintf(1, '.');
    end
    
    % NewtonDecrSqThreshold = realsqrt(eps); % Loose tolerance
    % NewtonDecrSqThreshold = 2*eps; % Tight tolerance
    NewtonDecrSqThreshold = 1e3*eps;
    if abs(NewtonDecrSq) < NewtonDecrSqThreshold
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
        rho = -1;                           % Force step size reduction and reject step
    else
        dm = -pg - 0.5*(z.'*z);             % Predicted reduction f - fp, where fp = f + p'*g + 0.5*p'*H*p
        rho = (f - fn)/dm;                  % Actual reduction divided by predicted reduction
    end
 
    if rho < 0.25
        lambda = min(lambda*11.0, 1e7);
    elseif rho > 0.75
        lambda = max(lambda/9.0, 1e-7);
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
            % [ Xi*p,              Xi ]
            % [    0, y^T/sqrt(y^T*p) ]
            RR = [ Xi*p, Xi ;
                   0, y.'/realsqrt(py)];
            % Q-less QR
            RR = qr(RR, "econ");
            Xi = RR(2:nx+1, 2:nx+1);
        end
    end
end
if verbosity > 1
    fprintf(2, 'WARNING: maximum number of iterations reached\n');
end
ret = 1;
return
