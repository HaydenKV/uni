% Minimise f(x) using trust-region quasi-Newton BFGS method (sparse square-root Hessian)
function [x, g, Xi, pidx, ret] = BFGSTrustSqrtSparse(costFunc, x, Xi, pidx, verbosity)

arguments
    costFunc (1, 1) function_handle
    x (:, 1) double
    Xi (:, :) double {mustBeSparse} = speye(length(x))
    pidx (1, :) double = 1:length(x)
    verbosity (1, 1) uint8 = 0
end

assert(istriu(Xi));

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
    p = funcmin.trsSqrtSparse(Xi, pidx, g, Delta);    % minimise 0.5*p.'*Pi*Xi.'*Xi*Pi.'*p + g.'*p subject to ||Xi*Pi.'*p|| <= Delta
    z = Xi*p(pidx); % Xi*Pi.'*p = Xi*I(:, pidx).'*p = Xi*I(pidx, :)*p = Xi*p(pidx)
    
    pg = p.'*g;
    LambdaSq = -pg;  % The Newton decrement squared is g.'*inv(H)*g = p.'*H*p = p.'*Pi*Xi.'*Xi*Pi.'*p
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
            % Construct inverse permutation pidxinv such that pidxinv(pidx) = 1:nx
            pidxinv = zeros(size(pidx));
            pidxinv(pidx) = 1:nx;

            % Compute column-pivoting QR decomposition: [Xi*Pi.'*p; 0]*Pi1 = [Xi*Pi.'*p; 0]*I(:, p1) = Q1*RR1 = [Y1, Z1]*[R1; 0]
            % where p1 reduces fill-in of R1
            [C1, RR1, p1] = qr(sparse([Xi*p(pidx); 0]), [Xi(:, pidxinv); y.'/realsqrt(py)], "vector");

            % Compute column-pivoting QR decomposition: Z1.'*[Xi*Pi.'; y.'/realsqrt(py)]*Pi2 = Z1.'*[Xi*Pi.'; y.'/realsqrt(py)]*I(:, p2) = Q2*R2
            % where p2 reduces fill-in of R2
            [~, R2, p2] = qr(C1(2:nx+1, :), zeros(nx, 0), "vector");
            Xi = R2;
            pidx = p2;
        end
    end
end
if verbosity > 1
    fprintf(2, 'WARNING: maximum number of iterations reached\n');
end
ret = 1;
return
